#ifndef BAG_OF_VISUAL_WORDS_HPP
#define BAG_OF_VISUAL_WORDS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <filesystem>

class BagOfVisualWords
{
    private:
        std::string image_extension_;
        int dictionary_size_;
        std::string dataset_path_;
        std::vector<cv::Mat> descriptors_;
        std::vector<int> class_labels_;
        cv::Mat k_labels_, k_centers_;
        cv::Mat findHistogram(cv::Mat descriptors);
        cv::Ptr<cv::ml::SVM> svm_;

    public:
        BagOfVisualWords();
        BagOfVisualWords(const std::string& dataset_path, const std::string& image_extension, const int dictionary_size);
        int processImagesClass(const std::string& class_name, const int class_label);
        int kMeansClustering(int attempts = 5, int iterations = 1e4, const std::string& file_path = "../kmeans_data.yml");
        int importKMeans(const std::string& file_path = "../kmeans_data.yml");
        int trainSVM(const std::string& file_path = "../svm_model.yml");
        int importSVM(const std::string& file_path = "../svm_model.yml");
        int predictImage(const std::string& file_path);
        int processImages();
        int runFullPipeline();
};

#endif