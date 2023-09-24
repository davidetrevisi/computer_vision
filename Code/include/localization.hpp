/*
    Author: Biffis Nicola
*/

#ifndef LOCALIZATION_HPP
#define LOCALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <iostream>
#include <fstream>

#include "bounding_box.hpp"
#include "segmentation.hpp"
#include "evaluation_metrics.hpp"

class Localization
{
    private:

        std::string pb;
        std::string pbt;
        std::vector<cv::String> images;
        std::vector<cv::String> truths;
        std::vector<cv::String> masks;
        
    public:
        Segmentation segmentation;
        EvaluationMetrics evaluation_metrics;
        cv::Mat background_mask;
        Localization();
        Localization(std::vector<cv::String>& images, std::vector<cv::String>& truths, std::vector<cv::String>& masks);
        ~Localization();
        void localization_function();
        void crop_function(cv::Mat& detection_image, cv::Mat& detectionMat, cv::Mat& image, int num, int j);
};

#endif