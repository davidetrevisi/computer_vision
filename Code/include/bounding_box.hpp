#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP

#include <vector>

class BoundingBox
{
    public:
        int bb_x, bb_y, bb_width, bb_height, bb_teamid;
        BoundingBox();
        BoundingBox(const int bb_x, const int bb_y, const int bb_width, const int bb_height, const int bb_teamid);
};

#endif