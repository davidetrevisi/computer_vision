/*
    Author: Biffis Nicola
*/

#include "localization.hpp"

int main(int argc, char** argv)
{
    std::vector<cv::String> images;
    std::vector<cv::String> truths;
    std::vector<cv::String> masks;

    // Path of the folder containing checkerboard images
    std::string path = "../Datasets/Testing/Images/*.jpg";
    std::string truths_path = "../Datasets/Testing/Masks/*.txt";
    std::string mask_path = "../Datasets/Testing/Masks/im*_bin.png";

    // Vectorize the files
    cv::glob(path, images);
    cv::glob(truths_path, truths);
    cv::glob(mask_path, masks);

    // Initialize the function
    Localization localization(images, truths, masks);

    // Start the detection
    localization.localization_function();

    return 0;
}