#include "evaluation_metrics.hpp"

using namespace std;
using namespace cv;

EvaluationMetrics::EvaluationMetrics() {
    file_path = "../Datasets/Testing/Masks/";
    iou_threshold = 0.5f;
    ground_truth_size = 0;

    gt_background = 0;
    gt_team_1 = 0;
    gt_team_2 = 0;
    gt_field = 0;

    players.clear();
}

void EvaluationMetrics::reset() {
    file_path = "../Datasets/Testing/Masks/";
    iou_threshold = 0.5f;
    ground_truth_size = 0;

    gt_background = 0;
    gt_team_1 = 0;
    gt_team_2 = 0;
    gt_field = 0;

    players.clear();
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
    //bb_file.open(file_path + file_name);
    bb_file.open(file_name);

    vector<int> data;

    int n;
    while (bb_file >> n) {
        data.push_back(n);
    }

    bb_file.close();

    ground_truth_size = data.size() / 5;

    BoundingBox bb_temp;
    int index = 0;
    float min_distance_value;

    for (int p = 0; p < players.size(); p++) {
        if (p > ground_truth_size) {
            players[p].ground_truth = BoundingBox();
        }
        else {
            index = 0;
            min_distance_value = 1000;
            bb_temp = BoundingBox();

            for (int i = 0; i < data.size(); i = i + 5) {
                float distance = rectangleDistance(data[i], data[i + 1], data[i + 2], data[i + 3], players[p].bounding_box.bb_x, players[p].bounding_box.bb_y, players[p].bounding_box.bb_width, players[p].bounding_box.bb_height);

                //cout << "GT: " << p << " " << distance << " " << min_distance_value << endl;
                if (distance < min_distance_value) {
                    bb_temp = BoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4]);
                    index = i;
                    min_distance_value = distance;
                }
            }
            // cout << data.size() << " " << index << endl;
            players[p].ground_truth = bb_temp;
            data.erase(data.begin() + index, data.begin() + index + 5);
        }
    }
}

void EvaluationMetrics::reversePlayers() {
    for (int p = 0; p < players.size(); p++) {
        if (players[p].bounding_box.bb_teamid == CLASS_TEAM_1) {
            players[p].bounding_box.bb_teamid = CLASS_TEAM_2;
        }
        else {
            players[p].bounding_box.bb_teamid = CLASS_TEAM_1;
        }
    }
}

float EvaluationMetrics::averagePrecision(const vector<float>& precision_values) {
    float sum = 0;

    for (int i = 0; i < 11; i++) {
        if (i < precision_values.size()) {
            //cout << precision_values[i] << endl;
            sum += precision_values[i];
        }
        else {
            sum += 0;
        }
    }
    //cout << sum << endl;

    float average_precision = (1.0f / 11.0f) * sum;
    return average_precision;
}

void EvaluationMetrics::computePrecisionRecall() {
    cumulative_fp = 0;
    cumulative_tp = 0;

    for (int i = 0; i < players.size(); i++) {
        if (players[i].intersectionOverUnion(iou_threshold)) {
            cumulative_tp++;
        }
        else {
            cumulative_fp++;
        }

        //cout << "Player: " << players[i].bounding_box.bb_x << endl;

        players[i].precision = (float)cumulative_tp / (cumulative_tp + cumulative_fp);
        players[i].recall = (float)cumulative_tp / ground_truth_size;

        //cout << "M.A.P. : " << ground_truth_size << " " << cumulative_fp << " " << cumulative_tp << " " << players[i].precision << " " << players[i].recall << endl;
    }
}

float EvaluationMetrics::meanAveragePrecision(const int team) {
    vector<float> precision_values;
    float current_max;
    float recall_flag = 0;

    for (int i = 0; i < players.size(); i++) {
        if (players[i].bounding_box.bb_teamid == team) {
            if (players[i].recall == recall_flag) {
                continue;
            }
            else {
                recall_flag = 0;
            }

            current_max = players[i].precision;

            for (int j = i; j < players.size(); j++) {
                if (players[j].bounding_box.bb_teamid == team) {
                    if (players[i].recall == players[j].recall && players[i].precision > players[j].precision) {
                        recall_flag = players[i].recall;
                        //cout << "Recall flag " << recall_flag << endl;
                        // current_max = players[j].precision;
                    }
                    else if (players[j].precision > current_max) {
                        // cout << "Max 1 modificato: " << current_max << endl;
                        current_max = players[j].precision;
                    }
                }
            }

            if (recall_flag == 0)
            {
                //cout << "Max 1: " << current_max << " i: " << i << endl;
                precision_values.push_back(current_max);
            }
            else {
                if (i == players.size() - 1) {
                    //cout << "Max 1: " << current_max << " i: " << i << endl;
                    precision_values.push_back(0.0f);
                }
                else {
                    //cout << "Max 1 recall: " << current_max << " i: " << i << endl;
                    precision_values.push_back(current_max);
                }
            }
        }
    }

    return averagePrecision(precision_values);
}

void EvaluationMetrics::meanAveragePrecisionPipeline(const string& file_name) {
    cout << file_name << endl;
    initializePlayersGroundTruth(file_name);

    computePrecisionRecall();

    float res_1 = meanAveragePrecision(CLASS_TEAM_1);
    float res_2 = meanAveragePrecision(CLASS_TEAM_2);

    float mean_average_precision = (1.0f / 2.0f) * (res_1 + res_2);
    cout << "Average precision team 1: " << res_1 << endl;
    cout << "Average precision team 2: " << res_2 << endl;
    cout << "Mean average precision: " << mean_average_precision << endl;

    reversePlayers();

    res_1 = meanAveragePrecision(CLASS_TEAM_1);
    res_2 = meanAveragePrecision(CLASS_TEAM_2);

    float mean_average_precision_reverse = (1.0f / 2.0f) * (res_1 + res_2);
    cout << "Reversing teams..." << endl;
    cout << "Average precision team 1: " << res_1 << endl;
    cout << "Average precision team 2: " << res_2 << endl;
    cout << "Mean average precision: " << mean_average_precision_reverse << endl;

    if (mean_average_precision > mean_average_precision_reverse) {
        std::cout << "Final mean average precision: " << mean_average_precision << std::endl;
        reversePlayers();
    }
    else {
        std::cout << "Final mean average precision: " << mean_average_precision_reverse << std::endl;
    }
}

void EvaluationMetrics::initializeSegmentationGroundTruth(const std::string& file_name) {
    //Mat segmentation_ground_truth = imread(file_path + file_name);
    segmentation_ground_truth = imread(file_name, COLOR_BGR2GRAY);

    for (int i = 0; i < segmentation_ground_truth.rows; i++) {
        for (int j = 0; j < segmentation_ground_truth.cols; j++) {
            if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_BACKGROUND) {
                gt_background++;
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_TEAM_1) {
                gt_team_1++;
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_TEAM_2) {
                gt_team_2++;
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_FIELD) {
                gt_field++;
            }
        }
    }
}

void EvaluationMetrics::reverseSegmentation(Mat& mask) {
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == CLASS_TEAM_1) {
                mask.at<uchar>(i, j) = CLASS_TEAM_2;
            }
            else if (mask.at<uchar>(i, j) == CLASS_TEAM_2) {
                mask.at<uchar>(i, j) = CLASS_TEAM_1;
            }
        }
    }
}

float EvaluationMetrics::intersectionOverUnion(const Mat& mask) {
    int intersection_background = 0, segmented_background = 0;
    int intersection_team_1 = 0, segmented_team_1 = 0;
    int intersection_team_2 = 0, segmented_team_2 = 0;
    int intersection_field = 0, segmented_field = 0;

    for (int i = 0; i < segmentation_ground_truth.rows; i++) {
        for (int j = 0; j < segmentation_ground_truth.cols; j++) {
            if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_BACKGROUND) {
                //cout << i << " " << j << endl;

                if (mask.at<uchar>(i, j) == 0) {
                    intersection_background++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_TEAM_1) {
                if (mask.at<uchar>(i, j) == 1) {
                    intersection_team_1++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_TEAM_2) {
                if (mask.at<uchar>(i, j) == 2) {
                    intersection_team_2++;
                }
            }
            else if (segmentation_ground_truth.at<uchar>(i, j) == CLASS_FIELD) {
                if (mask.at<uchar>(i, j) == 3) {
                    intersection_field++;
                }
            }
        }
    }

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == CLASS_BACKGROUND) {
                segmented_background++;
            }
            else if (mask.at<uchar>(i, j) == CLASS_TEAM_1) {
                segmented_team_1++;
            }
            else if (mask.at<uchar>(i, j) == CLASS_TEAM_2) {
                segmented_team_2++;
            }
            else if (mask.at<uchar>(i, j) == CLASS_FIELD) {
                segmented_field++;
            }
        }
    }

    //cout << intersection_background << " " << segmented_background << " " << gt_background << endl;
    float iou_background = (float)intersection_background / ((segmented_background + gt_background) - intersection_background);
    float iou_team_1 = (float)intersection_team_1 / ((segmented_team_1 + gt_team_1) - intersection_team_1);
    float iou_team_2 = (float)intersection_team_2 / ((segmented_team_2 + gt_team_2) - intersection_team_2);
    float iou_field = (float)intersection_field / ((segmented_field + gt_field) - intersection_field);

    float mean_iou = (1.0f / 4.0f) * (iou_background + iou_team_1 + iou_team_2 + iou_field);
    return mean_iou;

    return 0;
}

void EvaluationMetrics::segmentationPipeline(const std::string& file_name, cv::Mat& segmented_image) {
    initializeSegmentationGroundTruth(file_name);

    float res_1_iou = intersectionOverUnion(segmented_image);
    std::cout << "Intersection Over Union first try: " << res_1_iou << std::endl;

    reverseSegmentation(segmented_image);

    float res_2_iou = intersectionOverUnion(segmented_image);
    std::cout << "Intersection Over Union swapped: " << res_2_iou << std::endl;

    if (res_1_iou > res_2_iou) {
        std::cout << "Final Intersection Over Union: " << res_1_iou << std::endl;
        reverseSegmentation(segmented_image);
    }
    else {
        std::cout << "Final Intersection Over Union: " << res_2_iou << std::endl;
    }
}