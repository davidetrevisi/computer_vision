/*
    Author: Pavan Stefano
*/

#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP

#include <vector>

class BoundingBox
{
    public:
        int bb_x, bb_y, bb_width, bb_height, bb_teamid;
        float confidence;
        BoundingBox();
        BoundingBox(const int bb_x, const int bb_y, const int bb_width, const int bb_height, const int bb_teamid, const float confidence);
        ~BoundingBox();
};

#endif