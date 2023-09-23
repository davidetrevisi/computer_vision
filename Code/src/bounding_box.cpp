#include "bounding_box.hpp"

using namespace std;

/*
    Default constructor for the class
*/
BoundingBox::BoundingBox() {
    this->bb_x = 0;
    this->bb_y = 0;
    this->bb_width = 0;
    this->bb_height = 0;
    this->bb_teamid = 0;
    this->confidence = 0;
}

/*
    Constructor for the class with parameters
*/
BoundingBox::BoundingBox(const int bb_x, const int bb_y, const int bb_width, const int bb_height, const int bb_teamid, const float confidence) {
    this->bb_x = bb_x;
    this->bb_y = bb_y;
    this->bb_width = bb_width;
    this->bb_height = bb_height;
    this->bb_teamid = bb_teamid;
    this->confidence = confidence;
}

/*
    Destructor for the class
*/
BoundingBox::~BoundingBox() {
}