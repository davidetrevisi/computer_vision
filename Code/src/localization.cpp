#include "localization.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

Localization::Localization() {
    segmentation = Segmentation();
    evaluation_metrics = EvaluationMetrics();
    background_mask = Mat();

    //definisco le variabili per il modello da usare
    pb = "../localization/model/frozen_graph.pb";
    pbt = "../localization/model/frozen_graph.pbtxt";
}

Localization::Localization(std::vector<cv::String>& images, std::vector<cv::String>& truths, std::vector<cv::String>& masks) {
    segmentation = Segmentation();
    evaluation_metrics = EvaluationMetrics();
    background_mask = Mat();

    this->images = images;
    this->truths = truths;
    this->masks = masks;

    //definisco le variabili per il modello da usare
    pb = "../localization/model/frozen_graph.pb";
    pbt = "../localization/model/frozen_graph.pbtxt";
}

void Localization::localization_function() {
    // load the neural network model
    auto model = readNet(pb, pbt);

    Mat image;

    std::cout << "Numero di immagini da elaborare: " << images.size() << std::endl;

    for (int j = 0; j < images.size(); j++) {
        segmentation.reset();
        evaluation_metrics.reset();

        image = imread(images[j]);

        int num = 0;

        std::cout << "Elaborazione immagine " << j + 1 << ":" << endl;

        cout << "Segmenting the background..." << endl;
        background_mask = segmentation.backgroundSegmentationPipeline(image);
        cout << "Done!" << endl;
        //cout << "Fin qui" << endl;
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

        cout << "Assigning teams..." << endl;
        segmentation.teamId();
        cout << "Done!" << endl;

        cout << "Computing Mean Average Precision..." << endl;
        for (int i = 0; i < segmentation.player_colors.size(); i++) {
            evaluation_metrics.players.push_back(Player(segmentation.player_colors[i].first));
            //cout << segmentation.player_colors[i].first.bb_teamid << endl;
        }

        evaluation_metrics.meanAveragePrecisionPipeline(truths[j]);

        cout << masks[j] << endl;

        //cv::namedWindow("Ground Truth", WINDOW_NORMAL);
        //cv::imshow("Ground Truth", imread(masks[j], COLOR_BGR2GRAY));
        //cv::waitKey(0);

        cout << "Computing Intersection Over Union..." << endl;
        segmentation.addPlayerClass();

        cv::Mat final_image = segmentation.finalImage(detection_image);

        //cv::namedWindow("Final segmentation", WINDOW_NORMAL);
        //cv::imshow("Final segmentation", segmentation.segmented_image);
        //cv::waitKey(0);s

        evaluation_metrics.segmentationPipeline(masks[j], segmentation.segmented_image);

        std::cout << "All processes completed!" << std::endl;
        std::cout << "------------------------------" << std::endl;
        destroyAllWindows();
    };

}

void Localization::crop_function(Mat& detection_image, Mat& detectionMat, Mat& image, int num, int j) {

    //for used to evaluate all the detections
    for (int i = 0; i < detectionMat.rows; i++) {

        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.4 && class_id == 0) { //class_id = 0 -> person

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

            Mat player = segmentation.playersSegmentationPipeline(image, background_mask, Rect(box_x, box_y, box_width, box_height));

            BoundingBox bb = BoundingBox(box_x, box_y, box_width, box_height, 10 + i);
            segmentation.addPlayer(player, bb);
        }
    }

}