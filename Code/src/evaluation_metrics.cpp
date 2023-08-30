#include "evaluation_metrics.hpp"

using namespace std;
using namespace cv;

EvaluationMetrics::EvaluationMetrics() {
    file_path = "../Datasets/Testing/Masks/";
    iou_threshold = 0.5f;
    ground_truth_size = 0;
}

float EvaluationMetrics::rectangleDistance(const int a_1, const int b_1, const int c_1, const int d_1, const int a_2, const int b_2, const int c_2, const int d_2) {
    float center_1_x = (float)(a_1 + c_1) / 2.0f;
    float center_1_y = (float)(b_1 + d_1) / 2.0f;

    float center_2_x = (float)(a_2 + c_2) / 2.0f;
    float center_2_y = (float)(b_2 + d_2) / 2.0f;

    float distance_x = std::abs(center_1_x - center_2_x);
    float distance_y = std::abs(center_1_y - center_2_y);

    float distance = sqrtf(distance_x * distance_x + distance_y * distance_y);

    return distance;
}

void EvaluationMetrics::initializePlayersGroundTruth(const string& file_name) {
    std::ifstream bb_file;
    bb_file.open(file_path + file_name);

    vector<int> data;

    int n;
    while (bb_file >> n) {
        data.push_back(n);
    }

    bb_file.close();

    ground_truth_size = data.size() / 5;

    int index = 0;
    BoundingBox temp;
    float min_distance_value;

    for (int p = 0; p < players.size(); p++) {
        min_distance_value = 1000;
        temp = BoundingBox();

        for (int i = 0; i < data.size(); i = i + 5) {
            float distance = rectangleDistance(data[i], data[i + 1], data[i + 2], data[i + 3], players[p].bounding_box.bb_x, players[p].bounding_box.bb_y, players[p].bounding_box.bb_width, players[p].bounding_box.bb_height);

            //cout << p << " " << distance << " " << min_distance_value << endl;
            if (distance < min_distance_value) {
                temp = BoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4]);
                min_distance_value = distance;
            }
        }
        //cout << temp.bb_x << endl;
        players[p].ground_truth = temp;
    }
}

float EvaluationMetrics::averagePrecision(const vector<float>& precision_values) {
    float sum = 0;

    for (int i = 0; i < 11; i++) {
        if (i < precision_values.size()) {
            sum += precision_values[i];
        }
        else {
            sum += 0;
        }
    }

    float average_precision = 1 / 11 * sum;
    return average_precision;
}

float EvaluationMetrics::meanAveragePrecision(const int team) {
    for (int i = 0; i < players.size(); i++) {
        if (players[i].intersectionOverUnion(iou_threshold)) {
            cumulative_tp++;
        }
        else {
            cumulative_fp++;
        }

        players[i].precision = (float)cumulative_tp / (cumulative_tp + cumulative_fp);
        players[i].recall = (float)cumulative_tp / ground_truth_size;

        //cout << ground_truth_size << " " << cumulative_fp << " " << cumulative_tp << " " << players[i].precision << " " << players[i].recall << endl;
    }

    vector<float> precision_values_1, precision_values_2;
    float current_max;

    for (int i = 0; i < players.size(); i++) {
        if (players[i].bounding_box.bb_teamid == team) {
            current_max = players[i].precision;

            for (int j = i; j < players.size(); j++) {
                if (players[i].bounding_box.bb_teamid == team && players[j].precision > current_max) {
                    current_max = players[j].precision;
                }
            }

            precision_values_1.push_back(current_max);
        }
        else {
            current_max = players[i].precision;

            for (int j = i; j < players.size(); j++) {
                if (players[i].bounding_box.bb_teamid != team && players[j].precision > current_max) {
                    current_max = players[j].precision;
                }
            }

            precision_values_2.push_back(current_max);
        }
    }

    float mean_average_precision = 1 / 2 * (averagePrecision(precision_values_1) + averagePrecision(precision_values_2));
    return mean_average_precision;
}

float EvaluationMetrics::initializeSegmentationGroundTruth(const std::string& file_name, const int team_1, const int team_2) {
    Mat segmentation_ground_truth = imread(file_path + file_name);
    Mat segmented_image;

    int intersection_background = 0, gt_background = 0, segmented_background = 0;
    int intersection_team_1 = 0, gt_team_1 = 0, segmented_team_1 = 0;
    int intersection_team_2 = 0, gt_team_2 = 0, segmented_team_2 = 0;
    int intersection_field = 0, gt_field = 0, segmented_field = 0;

    for (int i = 0; i < segmentation_ground_truth.rows; i++) {
        for (int j = 0; j < segmentation_ground_truth.cols; j++) {
            if (segmentation_ground_truth.at<uchar>(j, i) == 0) {
                gt_background++;

                if (segmented_image.at<uchar>(j, i) == 0) {
                    intersection_background++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(j, i) == team_1) {
                gt_team_1++;

                if (segmented_image.at<uchar>(j, i) == team_1) {
                    intersection_team_1++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(j, i) == team_2) {
                gt_team_2++;

                if (segmented_image.at<uchar>(j, i) == team_2) {
                    intersection_team_2++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(j, i) == 3) {
                gt_field++;

                if (segmented_image.at<uchar>(j, i) == 3) {
                    intersection_field++;
                }
            }
        }
    }

    for (int i = 0; i < segmented_image.rows; i++) {
        for (int j = 0; j < segmented_image.cols; j++) {
            if (segmented_image.at<uchar>(j, i) == 0) {
                segmented_background++;
            }
            else if (segmented_image.at<uchar>(j, i) == team_1) {
                segmented_team_1++;
            }
            else if (segmented_image.at<uchar>(j, i) == team_2) {
                segmented_team_2++;
            }
            else if (segmented_image.at<uchar>(j, i) == 3) {
                segmented_field++;
            }
        }
    }

    float iou_background = (float)intersection_background / ((segmented_background + gt_background) - intersection_background);
    float iou_team_1 = (float)intersection_team_1 / ((segmented_team_1 + gt_team_1) - intersection_team_1);
    float iou_team_2 = (float)intersection_team_2 / ((segmented_team_2 + gt_team_2) - intersection_team_2);
    float iou_field = (float)intersection_field / ((segmented_field + gt_field) - intersection_field);

    float mean_iou = 1 / 4 * (iou_background + iou_team_1 + iou_team_2 + iou_field);
    return mean_iou;
}

float EvaluationMetrics::intersectionOverUnion(Mat& mask, Mat& gt_mask, const int object) {

    return 0;
}