#include "player.hpp"

using namespace std;
using namespace cv;

Player::Player() {
    bounding_box = BoundingBox();
}

Player::Player(BoundingBox& bb) {
    bounding_box = bb;
}

bool Player::intersectionOverUnion(const float iou_threshold) {
    Rect rect_bb = Rect(bounding_box.bb_x, bounding_box.bb_y, bounding_box.bb_width, bounding_box.bb_height);
    Rect rect_gt = Rect(ground_truth.bb_x, ground_truth.bb_y, ground_truth.bb_width, ground_truth.bb_height);

    Rect intersection = rect_bb & rect_gt;
    float rect_union = (rect_bb.area() + rect_gt.area()) - intersection.area();

    intersection_over_union = intersection.area() / rect_union;

    //cout << "Rect: " << intersection_over_union << " " << rect_union << " " << intersection.area() << endl;

    if (intersection_over_union < iou_threshold) {
        return false;
    }
    else {
        return true;
    }
}