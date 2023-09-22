#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <map>

#include "bounding_box.hpp"

#define KMEANS_BACKGROUND 6
#define KMEANS_PLAYERS 4

#define CLASS_BACKGROUND 0
#define CLASS_TEAM_1 1
#define CLASS_TEAM_2 2
#define CLASS_FIELD 3

struct Comparator
{
    bool operator()(const cv::Vec3b& left, const cv::Vec3b& right) const {
        return (left[0] != right[0]) ? (left[0] < right[0]) : ((left[1] != right[1]) ? (left[1] < right[1]) : (left[2] < right[2]));
    }
};

class Segmentation
{
    public:
        cv::Mat segmented_image;
        std::map<cv::Vec3b, int, Comparator> pixels;
        std::vector<std::pair<BoundingBox, cv::Vec3b>> player_colors;
        float area;
        Segmentation();
        void reset();
        cv::Mat3b quantize(cv::Mat3b& image, const int K);
        void getColors(cv::Mat3b& image);
        std::vector<std::pair<cv::Vec3b, int>> sortByCount();
        std::vector<cv::Vec3b> colorChoice();
        cv::Mat backgroundMask(cv::Mat3b& image);
        cv::Mat backgroundSegmentationPipeline(cv::Mat& image);
        cv::Mat playerSegmentation(cv::Mat& image, cv::Rect r);
        cv::Mat removeBackground(cv::Mat& players, cv::Mat& background);
        cv::Mat playersSegmentationPipeline(cv::Mat& image, cv::Mat& background, cv::Rect bb);
        void addSegmentedPlayer(cv::Mat& image, BoundingBox& bb);
        void addPlayerClass();
        void addPlayer(cv::Mat& image, BoundingBox& bb);
        void teamId(const int index = 0);
        void addSegmentedClass(cv::Mat& image);
        void finalImage(cv::Mat& bb_image);
};

#endif