#ifndef EVALUATION_METRICS_HPP
#define EVALUATION_METRICS_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include "bounding_box.hpp"
#include "player.hpp"

#define CLASS_BACKGROUND 0
#define CLASS_TEAM_1 1
#define CLASS_TEAM_2 2
#define CLASS_FIELD 3

class EvaluationMetrics
{
    private:
        std::string file_path;
        float iou_threshold;
        int ground_truth_size;
        cv::Mat segmentation_ground_truth;
        int gt_background, gt_team_1, gt_team_2, gt_field;

    public:
        std::vector<Player> players;
        int cumulative_tp, cumulative_fp;
        EvaluationMetrics();
        void reset();
        float rectangleDistance(const int a_1, const int b_1, const int c_1, const int d_1, const int a_2, const int b_2, const int c_2, const int d_2);
        void initializePlayersGroundTruth(const std::string& file_name = "im1_bb.txt");
        void reversePlayers();
        float averagePrecision(const std::vector<float>& precision_values);
        void computePrecisionRecall();
        float meanAveragePrecision(const int team);
        void meanAveragePrecisionPipeline(const std::string& file_name = "im1_bb.txt");
        void initializeSegmentationGroundTruth(const std::string& file_name = "im1_bin.png");
        void reverseSegmentation(cv::Mat& mask);
        float intersectionOverUnion(const cv::Mat& mask);
        void segmentationPipeline(const std::string& file_name, cv::Mat& segmented_image);
};

#endif