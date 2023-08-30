#include "player.hpp"

using namespace std;
using namespace cv;

Player::Player() {
    bounding_box = BoundingBox();
    this->confidence = 0;
}

Player::Player(const int bb_x, const int bb_y, const int bb_width, const int bb_height, const int bb_teamid, const float confidence) {
    bounding_box = BoundingBox(bb_x, bb_y, bb_width, bb_height, bb_teamid);
    this->confidence = confidence;
}

bool Player::intersectionOverUnion(const float iou_threshold) {
    Rect rect_bb = Rect(bounding_box.bb_x, bounding_box.bb_y, bounding_box.bb_width, bounding_box.bb_height);
    Rect rect_gt = Rect(ground_truth.bb_x, ground_truth.bb_y, ground_truth.bb_width, ground_truth.bb_height);

    Rect intersection = rect_bb & rect_gt;
    float rect_union = (rect_bb.area() + rect_gt.area()) - intersection.area();

    intersection_over_union = intersection.area() / rect_union;

    if (intersection_over_union < iou_threshold) {
        return false;
    }
    else {
        return true;
    }
}