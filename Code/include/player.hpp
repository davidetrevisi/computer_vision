/*
    Author: Trevisi Davide
*/

#ifndef PLAYER_HPP
#define PLAYER_HPP

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include "bounding_box.hpp"

class Player
{
    public:
        BoundingBox ground_truth, bounding_box;
        float intersection_over_union, precision, recall;
        Player();
        Player(BoundingBox& bb);
        ~Player();
        bool intersectionOverUnion(const float iou_threshold);
};

#endif