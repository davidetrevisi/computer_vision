#include "bounding_box.hpp"

using namespace std;

BoundingBox::BoundingBox() {
    this->bb_x = 0;
    this->bb_y = 0;
    this->bb_width = 0;
    this->bb_height = 0;
    this->bb_teamid = 0;
}

BoundingBox::BoundingBox(const int bb_x, const int bb_y, const int bb_width, const int bb_height, const int bb_teamid) {
    this->bb_x = bb_x;
    this->bb_y = bb_y;
    this->bb_width = bb_width;
    this->bb_height = bb_height;
    this->bb_teamid = bb_teamid;
}