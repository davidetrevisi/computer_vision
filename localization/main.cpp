#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
  
using namespace std;
using namespace cv;
using namespace dnn;

int main(int argc, char** argv)
{

    std::vector<std::string> class_names;
    ifstream ifs(string("names.txt").c_str());
    string line;

    while (getline(ifs, line))
    {   
        class_names.push_back(line);
    }

    //definisco le variabili per il modello da usare
    string pb  = "model/frozen_inference_graph.pb";
    string pbt = "model/config.pbtxt";

    // load the neural network model
    auto model = readNet(pb, pbt);

    //CICLO PER LEGGERE TUTTA LA CARTELLA

    std::vector<String> images;

    // Path of the folder containing checkerboard images
    std::string path = "../Datasets/Testing/Images/*.jpg";
 
    glob(path, images);

    Mat image;

    std::cout<<"Numero di immagini da elaborare: "<<images.size()<<std::endl;

    for(int j=0; j<images.size(); j++){

        image = imread(images[j]);
        
        int num = 0;

        std::cout<<"Elaborazione immagine "<<j+1<<": ";

        //create blob from image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);

        //create blob from image
        model.setInput(blob);

        //forward pass through the model to carry out the detection
        Mat output = model.forward();
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        //immagine dove salalvo le bb e il nome per mantenere quella originale "pulita" per la segmentazione
        Mat detection_image = image.clone();
        
        //for used to evaluate all the detections
        for (int i = 0; i < detectionMat.rows; i++){
            
            int class_id = detectionMat.at<float>(i, 1);
            float confidence = detectionMat.at<float>(i, 2);

            // Check if the detection is of good quality
            if (confidence > 0.4 && class_id==0){ //class_id = 0 -> humans
                
                //calcolo dei limiti delle bb
                int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
                int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);

                //disegno le bb
                rectangle(detection_image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(0,0,0), 2);

                num = num + 1;
                string txt = "Player "+to_string(num);

                //nomino le bb
                //putText(image, class_names[class_id-1].c_str(), Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
                putText(detection_image, txt, Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1);

                if(j==1){//ciclo provvisorio per vedere le sezioni di immagine su cui lavorare con la segmentazione 

                    Mat cropped_image = image(Range(box_y, box_y+box_height), Range(box_x, box_x+box_width));
                    imshow("Cropped Image", cropped_image);
                    waitKey(0);
            
                }
            
            }
       
        }

        std::cout<<"Done"<<std::endl;
        
        //nome contenente il path dove salvare le immagini
        string name = "../Datasets/Results/im"+to_string(j+1)+".jpg";
       
        imwrite(name, detection_image);
        //imshow("image", image);
        //waitKey(0);
        destroyAllWindows();
    };  

    return (0);
}


//https://www.kaggle.com/datasets/sovitrath/sports-image-dataset DATASET
//https://github.com/rishavgiri6/Sporting_Video_Classifier/tree/master TRAINING KERAS (CLASSIFICAZIONE NON LOCALIZZAZIONE)

//https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API MODELLI GIA TRAINATI 
