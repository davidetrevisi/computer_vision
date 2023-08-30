#ifndef EVALUATION_METRICS_HPP
#define EVALUATION_METRICS_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include "bounding_box.hpp"
#include "player.hpp"

class EvaluationMetrics
{
    private:
        std::string file_path;
        float iou_threshold;
        int ground_truth_size;

    public:
        std::vector<Player> players;
        int cumulative_tp, cumulative_fp;
        EvaluationMetrics();
        float rectangleDistance(const int a_1, const int b_1, const int c_1, const int d_1, const int a_2, const int b_2, const int c_2, const int d_2);
        void initializePlayersGroundTruth(const std::string& file_name = "im1_bb.txt");
        float initializeSegmentationGroundTruth(const std::string& file_name = "im1_bin.png", const int team_1, const int team_2);
        float averagePrecision(const std::vector<float>& precision_values);
        float meanAveragePrecision(const int team);
        float intersectionOverUnion(cv::Mat& mask, cv::Mat& gt_mask, const int object);
};

#endif