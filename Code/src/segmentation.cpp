#include "segmentation.hpp"

using namespace cv;
using namespace std;

/*
    Default constructor for the class
*/
Segmentation::Segmentation() {
    area = 0;
    pixels.clear();
    player_colors.clear();
    segmented_image = Mat();
}

/*
    Destructor for the class
*/
Segmentation::~Segmentation() {
}

/*
    Function that sets the variable to default value
*/
void Segmentation::reset() {
    area = 0;
    pixels.clear();
    player_colors.clear();
    segmented_image = Mat();
}

/*
    Function that performs the quantization using KMeans
    on the input image
*/
Mat3b Segmentation::quantize(Mat3b& image, const int K) {
    pixels.clear();

    // Prepare the image by reshaping it
    area = image.rows * image.cols;
    Mat data = image.reshape(1, area);
    data.convertTo(data, CV_32F);

    // Run KMeans
    vector<int> labels;
    Mat1f colors;
    kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

    for (int i = 0; i < area; ++i)
    {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }

    // Reshape and save the output
    Mat3b dst;
    Mat reduced = data.reshape(3, image.rows);
    reduced.convertTo(dst, CV_8U);

    //cv::imshow("Quantized", dst);
    //cv::waitKey(0);

    return dst;
}

/*
    Function that counts the occurrencies of the colors
    in the image
*/
void Segmentation::getColors(cv::Mat3b& image) {
    for (int r = 0; r < image.rows; ++r)
    {
        for (int c = 0; c < image.cols; ++c)
        {
            Vec3b color = image(r, c);
            if (pixels.count(color) == 0)
            {
                pixels[color] = 1;
            }
            else
            {
                pixels[color] = pixels[color] + 1;
            }
        }
    }

    /*
    for (auto color : pixels)
    {
        cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    }*/
}

/*
    Function that sorts the colors in 'pixels' by number of occurrencies
*/
std::vector<std::pair<Vec3b, int>> Segmentation::sortByCount() {
    std::vector<std::pair<Vec3b, int>> pairs(pixels.begin(), pixels.end());

    std::sort(pairs.begin(), pairs.end(), [](std::pair<cv::Vec3b, int>& first, std::pair<cv::Vec3b, int>& second) {
        return first.second > second.second;
        });

    return pairs;
}

/*
    Function that selects the most present color(s) in the image and returns it(them)
*/
std::vector<Vec3b> Segmentation::colorChoice() {
    // Sort the colors
    std::vector<std::pair<Vec3b, int>> colors = sortByCount();
    std::vector<Vec3b> output;

    // If the main color is black, erase it because the segmented images have black backgrounds
    if (colors[0].first[0] < 20 && colors[0].first[1] < 20 && colors[0].first[2] < 20) {
        colors.erase(colors.begin());
    }

    // Compute the percentage of coverage
    float percentage = 100.0f * (float)colors[0].second / area;
    float percentage_2 = 100.0f * (float)colors[1].second / area;
    
    /*for (auto color : colors)
    {
        //cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    }*/

    // Save the main color
    output.push_back(colors[0].first);

    // If close enough save also the second main color
    if (((percentage - percentage_2) <= 7.00f) || (abs(colors[0].first[0] - colors[1].first[0]) < 15 && abs(colors[0].first[1] - colors[1].first[1]) < 15 && abs(colors[0].first[2] - colors[1].first[2] < 15))) {
        output.push_back(colors[1].first);
    }

    return output;
}

/*
    Function that sets the pixels in the field mask
    to the appropriate class
*/
Mat Segmentation::fieldMask(cv::Mat3b& image) {
    getColors(image);
    vector<Vec3b> bg_colors = colorChoice();
    Mat field(image.rows, image.cols, CV_8U, Scalar(0));

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int c = 0; c < bg_colors.size(); c++) {
                if (image.at<Vec3b>(i, j) == bg_colors[c]) {
                    field.at<uchar>(i, j) = CLASS_FIELD;
                }
            }
        }
    }

    return field;
}

/*
    Function that executes the whole pipeline for the
    field segmentation
*/
cv::Mat Segmentation::fieldSegmentationPipeline(cv::Mat& image) {
    segmented_image = Mat(image.size(), CV_8U, Scalar(0));

    Mat3b input = image.clone();

    Mat3b reduced = quantize(input, KMEANS_BACKGROUND);
    Mat mask = fieldMask(reduced);

    addFieldClass(mask);

    return mask;
}

/*
    Function that segments the players using the GrabCut algorithm
*/
cv::Mat Segmentation::playerSegmentation(cv::Mat& image, cv::Rect r) {
    // Perform GrabCut segmentation
    cv::Mat mask(image.size(), CV_8UC1);
    cv::Mat bg, fg;
    cv::Mat result(image.size(), CV_8UC3);
    cv::grabCut(image, mask, r, bg, fg, 10, cv::GC_INIT_WITH_RECT);

    // Draw the segmentation
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            if ((int)mask.at<uchar>(row, col) == 2)        //BACKGROUND
            {
                result.at<cv::Vec3b>(row, col) = Vec3b(0, 0, 0);
            }
            else if ((int)mask.at<uchar>(row, col) == 3)        //FOREGROUND
            {
                result.at<cv::Vec3b>(row, col) = image.at<Vec3b>(row, col);
            }
        }
    }

    Mat cropped_image = result(r);

    //cv::imshow("Segmented Image", cropped_image);
    //cv::waitKey(0);

    return cropped_image;
}

/*
    Function that removes the field found before from the segmented player image
*/
cv::Mat Segmentation::removeBackground(cv::Mat& player, cv::Mat& background) {
    Mat output = player.clone();

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            if (background.at<uchar>(i, j) == CLASS_FIELD) {
                output.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
        }
    }

    //cv::imshow("Final player", output);
    //cv::waitKey(0);

    return output;
}

/*
    Function that executes the whole pipeline for the
    player segmentation
*/
cv::Mat Segmentation::playersSegmentationPipeline(cv::Mat& image, cv::Mat& background, cv::Rect bb) {
    Mat cropped_background = background(bb);
    Mat player = playerSegmentation(image, bb);

    Mat output = removeBackground(player, cropped_background);

    return output;
}

/*
    Function that paints the pixels with the segmented player to
    their corresponding color in the segmented image
*/
void Segmentation::addSegmentedPlayer(cv::Mat& image, BoundingBox& bb) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                segmented_image.at<uchar>(i + bb.bb_y, j + bb.bb_x) = bb.bb_teamid;
            }
        }
    }

    //cv::namedWindow("Seg", WINDOW_NORMAL);
    //cv::imshow("Seg", segmented_image);
    //cv::waitKey(0);
}

/*
    Function that paints the pixels of each player to their corresponding team id
    in the segmented image
*/
void Segmentation::addPlayerClass() {
    // Cycle from the end so players with highest confidence don't get overwritten
    for (int p = player_colors.size() - 1; p >= 0; p--) {
        // Crop the image
        Mat cropped = segmented_image(Rect(player_colors[p].first.bb_x, player_colors[p].first.bb_y, player_colors[p].first.bb_width, player_colors[p].first.bb_height));

        //cout << cropped.rows << " " << cropped.cols << " " << player_colors[p].first.bb_width << endl;

        // Assign the corresponding color
        if (player_colors[p].first.bb_teamid == CLASS_TEAM_1) {
            for (int i = 0; i < cropped.rows; i++) {
                for (int j = 0; j < cropped.cols; j++) {
                    if (cropped.at<uchar>(i, j) == 10 + p) {
                        segmented_image.at<uchar>(i + player_colors[p].first.bb_y, j + player_colors[p].first.bb_x) = CLASS_TEAM_1;
                    }
                }
            }
        }
        else if (player_colors[p].first.bb_teamid == CLASS_TEAM_2) {
            for (int i = 0; i < cropped.rows; i++) {
                for (int j = 0; j < cropped.cols; j++) {
                    if (cropped.at<uchar>(i, j) == 10 + p) {
                        segmented_image.at<uchar>(i + player_colors[p].first.bb_y, j + player_colors[p].first.bb_x) = CLASS_TEAM_2;
                    }
                }
            }
        }
    }
    //cv::namedWindow("Seg", WINDOW_NORMAL);
    //cv::imshow("Seg", segmented_image);
    //cv::waitKey(0);
}

/*
    Function that executes all the actions required to add
    a player to the segmented image
*/
void Segmentation::addPlayer(cv::Mat& image, BoundingBox& bb) {
    Mat3b input = image.clone();
    Mat3b reduced = quantize(input, KMEANS_PLAYERS);

    //cv::namedWindow("Input", WINDOW_NORMAL);
    //cv::imshow("Input", image);
    //cv::waitKey(0);

    getColors(reduced);
    vector<Vec3b> team_colors = colorChoice();

    player_colors.push_back(pair<BoundingBox, Vec3b>(bb, team_colors[0]));

    addSegmentedPlayer(image, bb);
}

/*
    Function that assigns the corresponding team id to each player
    based on the color
*/
void Segmentation::teamId(const int index) {
    player_colors[index].first.bb_teamid = CLASS_TEAM_1;
    Vec3b team_1_color = player_colors[0].second;
    Vec3b team_2_color;
    bool flag = true;

    for (int i = 0; i < player_colors.size(); i++) {
        // If the colors are close enough assign the team,
        // otherwise assign the other one
        if (abs(player_colors[i].second[0] - team_1_color[0]) < 40 && abs(player_colors[i].second[1] - team_1_color[1]) < 40 && abs(player_colors[i].second[2] - team_1_color[2]) < 40) {
            player_colors[i].first.bb_teamid = CLASS_TEAM_1;
        }
        else {
            if (flag) {
                team_2_color = player_colors[i].second;
                flag = false;
            }
            player_colors[i].first.bb_teamid = CLASS_TEAM_2;
        }
    }

    // Performs the same process again to make sure that
    // players in the second team are really together
    for (int i = 0; i < player_colors.size(); i++) {
        if (player_colors[i].first.bb_teamid == CLASS_TEAM_2) {
            if (abs(player_colors[i].second[0] - team_2_color[0]) < 40 && abs(player_colors[i].second[1] - team_2_color[1]) < 40 && abs(player_colors[i].second[2] - team_2_color[2]) < 40) {
                player_colors[i].first.bb_teamid = CLASS_TEAM_2;
            }
            else {
                player_colors[i].first.bb_teamid = 3;
            }
        }
    }
}

/*
    Function that adds the segmented field to the
    segmented image
*/
void Segmentation::addFieldClass(cv::Mat& image) {
    for (int i = 0; i < segmented_image.rows; i++) {
        for (int j = 0; j < segmented_image.cols; j++) {
            if (image.at<uchar>(i, j) != 0) {
                segmented_image.at<uchar>(i, j) = CLASS_FIELD;
            }
        }
    }

    //cv::namedWindow("Seg", WINDOW_NORMAL);
    //cv::imshow("Seg", segmented_image);
    //cv::waitKey(0);
}

/*
    Function that computes the final images to show the user
*/
void Segmentation::finalImage(cv::Mat& bb_image) {
    cv::Mat background;

    cv::cvtColor(bb_image, background, COLOR_BGR2BGRA);

    // Make sure all the pixels are assigned, otherwise assign them the background value
    for (int i = 0; i < segmented_image.rows; i++) {
        for (int j = 0; j < segmented_image.cols; j++) {
            if (segmented_image.at<uchar>(i, j) != CLASS_BACKGROUND && segmented_image.at<uchar>(i, j) != CLASS_TEAM_1 && segmented_image.at<uchar>(i, j) != CLASS_TEAM_2 && segmented_image.at<uchar>(i, j) != CLASS_FIELD) {
                segmented_image.at<uchar>(i, j) = CLASS_BACKGROUND;
            }
        }
    }

    //cv::namedWindow("Final segmentation", WINDOW_NORMAL);
    //cv::imshow("Final segmentation", segmented_image);
    //cv::waitKey(0);

    // Compute and show the segmented image with colors
    cv::Mat finale(segmented_image.size(), CV_8UC4);
    cv::Mat bgr_segmentation(segmented_image.size(), CV_8UC4);

    for (int i = 0; i < segmented_image.rows; i++) {
        for (int j = 0; j < segmented_image.cols; j++) {
            if (segmented_image.at<uchar>(i, j) == CLASS_TEAM_1) {
                bgr_segmentation.at<Vec4b>(i, j) = Vec4b(0, 0, 255, 1);
            }
            else if (segmented_image.at<uchar>(i, j) == CLASS_TEAM_2) {
                bgr_segmentation.at<Vec4b>(i, j) = Vec4b(0, 255, 0, 1);
            }
            else if (segmented_image.at<uchar>(i, j) == CLASS_FIELD) {
                bgr_segmentation.at<Vec4b>(i, j) = Vec4b(255, 0, 0, 1);
            }
            else {
                bgr_segmentation.at<Vec4b>(i, j) = Vec4b(0, 0, 0, 0);
            }
        }
    }

    cv::namedWindow("Coloured segmented image", WINDOW_NORMAL);
    cv::imshow("Coloured segmented image", bgr_segmentation);
    cv::waitKey(0);

    // Compute and show the final image made of the image
    // with the bounding boxes drawn and the previous one merged
    // with weights and using transparency
    addWeighted(background, 0.7, bgr_segmentation, 0.5, 0, finale);

    cv::Mat final_image;
    cv::cvtColor(finale, final_image, COLOR_BGRA2BGR);

    cv::namedWindow("Program output", WINDOW_NORMAL);
    cv::imshow("Program output", final_image);
    cv::waitKey(0);
}