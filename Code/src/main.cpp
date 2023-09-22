#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include "localization.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    std::vector<std::string> class_names;
    ifstream ifs(string("names.txt").c_str());
    string line;

    while (getline(ifs, line))
    {
        class_names.push_back(line);
    }

    //CICLO PER LEGGERE TUTTA LA CARTELLA

    std::vector<String> images;
    std::vector<String> truths;
    std::vector<String> masks;

    // Path of the folder containing checkerboard images
    std::string path = "../Datasets/Testing/Images/*.jpg";
    std::string truths_path = "../Datasets/Testing/Masks/*.txt";
    std::string mask_path = "../Datasets/Testing/Masks/im*_bin.png";

    glob(path, images);
    glob(truths_path, truths);
    glob(mask_path, masks);

    Localization localization = Localization(images, truths, masks);

    localization.localization_function();

    std::cout << "Done" << std::endl;

    return (0);
}