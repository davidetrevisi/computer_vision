#include "localization.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

/*
    Default constructor for the class
*/
Localization::Localization()
    : segmentation(Segmentation())
    , evaluation_metrics(EvaluationMetrics())
{
    // Initialize all the parameters and classes
    background_mask = Mat();

    pb = "../localization/model/frozen_graph.pb";
    pbt = "../localization/model/frozen_graph.pbtxt";
}

/*
    Destructor for the class
*/
Localization::~Localization() {
}

/*
    Constructor for the class with parameters:
        - vector of images
        - vector of paths to the files containing the bounding boxes ground truths
        - vector of paths to the files containing the segmentation ground truths
*/
Localization::Localization(vector<String>& images, vector<String>& truths, vector<String>& masks)
    : segmentation(Segmentation())
    , evaluation_metrics(EvaluationMetrics())
{
    // Initialize all the parameters and classes
    background_mask = Mat();

    this->images = images;
    this->truths = truths;
    this->masks = masks;

    pb = "../localization/model/frozen_graph.pb";
    pbt = "../localization/model/frozen_graph.pbtxt";
}

/*
    Function that processes all the images
*/
void Localization::localization_function() {
    // load the neural network model
    auto model = readNet(pb, pbt);

    Mat image;

    cout << "Numero di immagini da elaborare: " << images.size() << endl;

    for (int j = 3; j < images.size(); j++) {
        // Initialize to default the parameters of the other classes
        // to avoid errors when processing multiple images

        segmentation.reset();
        evaluation_metrics.reset();

        // Read the image
        image = imread(images[j]);

        if (image.empty()) {
            cerr << "[ERROR] Could not read the input image!" << endl;
            exit(1);
        }

        int num = 0;

        cout << "Elaborazione immagine " << j + 1 << ":" << endl;

        // Segment the field by calling the appropriate function
        cout << "Segmenting the field..." << endl;
        background_mask = segmentation.fieldSegmentationPipeline(image);
        cout << "Done!" << endl;

        //create blob from image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);

        //create blob from image
        model.setInput(blob);

        cout << "Finding and processing players (it may take some time)..." << endl;

        //forward pass through the model to carry out the detection
        Mat output = model.forward();

        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        //immagine dove salalvo le bb e il nome per mantenere quella originale "pulita" per la segmentazione
        Mat detection_image = image.clone();

        //FUNCTION THAT CROPS THE DETECTIONS, SAVING THEM IN DIFFERENT VARIABLES
        crop_function(detection_image, detectionMat, image, num, j);

        cout << "Done!" << endl;

        // Assign teams to players by calling the appropriate function
        cout << "Assigning teams..." << endl;
        segmentation.teamId();
        cout << "Done!" << endl;

        // Add the players found to the class that will evaluate the metrics
        for (int i = 0; i < segmentation.player_colors.size(); i++) {
            evaluation_metrics.players.push_back(Player(segmentation.player_colors[i].first));
            //cout << segmentation.player_colors[i].first.bb_teamid << endl;
        }

        // Evaluate Mean Average Precision by calling the appropriate function
        cout << "Computing Mean Average Precision..." << endl;
        evaluation_metrics.meanAveragePrecisionPipeline(truths[j]);
        cout << "Done!" << endl;

        //cout << masks[j] << endl;

        //namedWindow("Ground Truth", WINDOW_NORMAL);
        //imshow("Ground Truth", imread(masks[j], COLOR_BGR2GRAY));
        //waitKey(0);

        cout << "Adding players to the image..." << endl;
        segmentation.addPlayerClass();
        cout << "Done!" << endl;

        //namedWindow("Final segmentation", WINDOW_NORMAL);
        //imshow("Final segmentation", segmentation.segmented_image);
        //waitKey(0);

        // Evaluate Mean Intersection Over Union by calling the appropriate function
        cout << "Computing Intersection Over Union..." << endl;
        evaluation_metrics.segmentationPipeline(masks[j], segmentation.segmented_image);
        cout << "Done!" << endl;

        // Compute the final images and show them by calling the appropriate function

        segmentation.finalImage(detection_image);
        destroyAllWindows();

        cout << "All processes completed!" << endl;
        cout << "------------------------------" << endl;
    };
}

/*
    Function ...
*/
void Localization::crop_function(Mat& detection_image, Mat& detectionMat, Mat& image, int num, int j) {
    //for used to evaluate all the detections
    for (int i = 0; i < detectionMat.rows; i++) {

        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.5 && class_id == 0) { //class_id = 0 -> person

            //calcolo dei limiti delle bb
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);

            //disegno le bb
            rectangle(detection_image, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(0, 0, 0), 2);

            num = num + 1;
            string txt = "Player " + to_string(num);

            putText(detection_image, txt, Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

            Mat cropped_image = image(Range(box_y, box_y + box_height), Range(box_x, box_x + box_width));

            // Pass the original image, the segmented background and the bounding box
            // found to segment the player
            Mat player = segmentation.playersSegmentationPipeline(image, background_mask, Rect(box_x, box_y, box_width, box_height));

            // Add the player to the image for the segmentation
            BoundingBox bb = BoundingBox(box_x, box_y, box_width, box_height, 10 + num - 1, confidence);
            segmentation.addPlayer(player, bb);
        }
    }
}