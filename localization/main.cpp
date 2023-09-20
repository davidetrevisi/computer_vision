#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
  
using namespace std;
using namespace cv;
using namespace dnn;

void localization_function(std::vector<String> images);
void crop_function(std::vector<Mat> &detections, Mat &detection_image, Mat detectionMat, Mat image, int num, int j);

int main(int argc, char** argv)
{
    // Path of the folder containing checkerboard images
    std::string path = "../Datasets/Testing/Images/*.jpg";

    std::vector<String> images;

    glob(path, images);

    //FUNZIONE DI LOCALIZZAZIONE
    localization_function(images);

    return (0);
}

void localization_function(std::vector<String> images){

    //definisco le variabili per il modello da usare
    string pb  = "model/frozen_inference_graph.pb";
    string pbt = "model/config.pbtxt";

    // load the neural network model
    auto model = readNet(pb, pbt);
    
    Mat image;

    std::cout<<"Numero di immagini da elaborare: "<<images.size()<<std::endl;

    for(int j=0; j<images.size(); j++){

        image = imread(images[j]);
        
        int num = 0;

        std::cout<<"Elaborazione immagine "<<j+1<<" ... "<<endl;
 
        //create blob from image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);

        //create blob from image
        model.setInput(blob);

        //forward pass through the model to carry out the detection
        Mat output = model.forward();
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        //immagine dove salalvo le bb e il nome per mantenere quella originale "pulita" per la segmentazione
        Mat detection_image = image.clone();

        std::vector<Mat> detections;

        //FUNCTION THAT CROPS THE DETECTIONS, SAVING THEM IN DIFFERENT VARIABLES
        crop_function(detections, detection_image, detectionMat, image, num, j);

        std::cout<<"Giocatori individuati nell'immagine: "<<detections.size()<<endl;

        std::cout<<"Stato: Done"<<std::endl;
        std::cout<<"------------------------------"<<std::endl;
        
        //nome contenente il path dove salvare le immagini
        string name = "../Datasets/Results/im"+to_string(j+1)+".jpg";
       
        imwrite(name, detection_image);
    };

}

void crop_function(std::vector<Mat> &detections, Mat &detection_image, Mat detectionMat, Mat image, int num, int j){

    //for used to evaluate all the detections
    for (int i = 0; i < detectionMat.rows; i++){
            
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.4 && class_id==0){ //class_id = 0 -> person
                
            //calcolo dei limiti delle bb
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);

            //disegno le bb
            rectangle(detection_image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(0,0,0), 2);

            num = num + 1;
            string txt = "Player "+to_string(num);

            putText(detection_image, txt, Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1);

            Mat cropped_image = image(Range(box_y, box_y+box_height), Range(box_x, box_x+box_width));
            detections.push_back(cropped_image);
            
            string name = "../Datasets/Results/single_image/im"+to_string(j+1)+"_"+to_string(i+1)+".jpg";;

            imwrite(name, cropped_image); //saving every single cropped image to work on it
                     
        }
       
    }

}