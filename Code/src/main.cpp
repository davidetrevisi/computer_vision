// #include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

// #include <Windows.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>
// #include <omp.h>

#define NUM_THREADS 8

using namespace std;
using namespace cv;
using namespace cv::ml;

const string DATASET_PATH = "../Dataset/";
const string IMAGE_EXT = ".png";
const int TESTING_PERCENT_PER = 3;
const int DICT_SIZE = 115; // 80 word per class

// inline bool fileExists(const std::string &name)
//{
//     struct stat buffer;
//     return (stat(name.c_str(), &buffer) == 0);
// }

Mat allDescriptors;
vector<Mat> allDescPerImg;
vector<int> allClassPerImg;
int allDescPerImgNum = 0;
void readDetectComputeimage(const string &className, int imageNumbers, int classLable)
{
#pragma omp parallel
    {
#pragma omp for schedule(dynamic) ordered
        for (int i = 1; i <= imageNumbers; i++)
        {
            // If this image is test not use this in learning
            // if (i % TESTING_PERCENT_PER == 0)
            //{
            //    continue;
            //}
            ostringstream ss;
            Mat grayimg;
            Ptr<SIFT> siftptr;
            siftptr = SIFT::create();

            // Load image, Detect and Describe features
            ss.str("");
            ss << i;
            // if (fileExists(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT))
            //{
            // std::cout << DATASET_PATH + className + "/image_" + ss.str() + IMAGE_EXT << std::endl;
            cvtColor(imread(DATASET_PATH + className + "/image_" + ss.str() + IMAGE_EXT), grayimg, COLOR_BGR2GRAY);
            // std::cout << "Ciao" << std::endl;
            vector<KeyPoint> keypoints;
            Mat descriptors;
            siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);
#pragma omp critical
            {
                allDescriptors.push_back(descriptors);
                allDescPerImg.push_back(descriptors);
                allClassPerImg.push_back(classLable);
                allDescPerImgNum++;
            }

            /*Mat output;
            drawKeypoints(grayimg, keypoints, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imshow("Result [" + ss.str() + IMAGE_EXT + "]", output);
            waitKey(10);
            destroyWindow("Result [" + ss.str() + ".jpg]");*/
            //}
            // else
            //{
            //    break;
            //}
        }
    }
}

Mat kCenters, kLabels;
Mat getDataVector(Mat descriptors)
{
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors, kCenters, matches);

    // Make a Histogram of visual words
    Mat datai = Mat::zeros(1, DICT_SIZE, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++)
    {
        datai.at<float>(0, matches.at(index).trainIdx) = datai.at<float>(0, matches.at(index).trainIdx) + 1;
    }
    return datai;
}

Mat inputData;
Mat inputDataLables;
void getHistogram(const string &className, int imageNumbers, int classLable)
{
#pragma omp parallel
    {
#pragma omp for schedule(dynamic) ordered
        for (int i = 1; i <= imageNumbers; i++)
        {
            // If this image is test not use this in learning
            // if (i % TESTING_PERCENT_PER == 0)
            //{
            //    continue;
            //}
            ostringstream ss;
            Mat grayimg;
            Ptr<SIFT> siftptr;
            siftptr = SIFT::create();
            // Load image, Detect and Describe features
            ss.str("");
            ss << i;
            // if (fileExists(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT))
            //{
            cvtColor(imread(DATASET_PATH + className + "/image_" + ss.str() + IMAGE_EXT), grayimg, cv::COLOR_BGR2GRAY);
            vector<KeyPoint> keypoints;
            Mat descriptors;
            siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);

#pragma omp critical
            {
                inputData.push_back(getDataVector(descriptors));
                inputDataLables.push_back(Mat(1, 1, CV_32SC1, classLable));
            }
            //}
            // else
            //{
            //    break;
            //}
        }
    }
}

void getHistogramFast()
{
#pragma omp parallel
    {
#pragma omp for schedule(dynamic) ordered
        for (int i = 0; i < allDescPerImgNum; i++)
        {
            Mat dvec = getDataVector(allDescPerImg[i]);
#pragma omp critical
            {
                inputData.push_back(dvec);
                inputDataLables.push_back(Mat(1, 1, CV_32SC1, allClassPerImg[i]));
            }
        }
    }
}

Ptr<SVM> svm;
double testData(const string &className, int imageNumbers, int classLable)
{
    for (int i = TESTING_PERCENT_PER; i <= imageNumbers; i += TESTING_PERCENT_PER)
    {
        Mat grayimg;
        Ptr<SIFT> siftptr;
        float r = 0;
        siftptr = SIFT::create();
        // Load image, Detect and Describe features
        cvtColor(imread("../Test/image_1.png"), grayimg, COLOR_BGR2GRAY);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);
        Mat dvector = getDataVector(descriptors);

        if (svm->predict(dvector) == classLable)
        {
            std::cout << "Found " << classLable << std::endl;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    cout << "Object detector started." << endl;
    clock_t sTime = clock();
    cout << "Reading inputs..." << endl;
    readDetectComputeimage("grilled_pork_cutlet", 6, 6);
    readDetectComputeimage("fish_cutlet", 10, 7);
    readDetectComputeimage("beans", 14, 10);
    readDetectComputeimage("salad", 19, 12);
    readDetectComputeimage("bread", 12, 13);
    readDetectComputeimage("basil_potatoes", 14, 11);
    readDetectComputeimage("pasta_with_pesto", 3, 1);
    readDetectComputeimage("rabbit", 8, 8);
    readDetectComputeimage("pasta_with_tomato_sauce", 6, 2);
    readDetectComputeimage("pilaw_rice_with_peppers_and_peas", 4, 5);
    readDetectComputeimage("pasta_with_clams_and_mussels", 12, 4);
    readDetectComputeimage("pasta_with_meat_sauce", 2, 3);
    readDetectComputeimage("seafood_salad", 3, 9);
    cout << "-> Reading, Detect and Describe input in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

    int clusterCount = DICT_SIZE, attempts = 5, iterationNumber = 1e4;
    sTime = clock();
    cout << "Running kmeans..." << endl;
    kmeans(allDescriptors, clusterCount, kLabels, TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters);
    cout << "-> kmeans run in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

    /*FileStorage storage("kmeans-starfish,sunflower,crab,trilobite.yml", FileStorage::WRITE);
    storage << "kLabels" << kLabels << "kCenters" << kCenters;
    storage.release();*/

    sTime = clock();
    /*cout << "Loading kmeans data..." << endl;
    FileStorage storage("kmeans-starfish,sunflower,crab,trilobite.yml", FileStorage::READ);
    storage["kLabels"] >> kLabels;
    storage["kCenters"] >> kCenters;
    storage.release();
    cout << "-> kmeans data loaded in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;*/

    sTime = clock();
    cout << "Finding histograms..." << endl;
    getHistogram("grilled_pork_cutlet", 6, 6);
    getHistogram("fish_cutlet", 10, 7);
    getHistogram("beans", 14, 10);
    getHistogram("salad", 19, 12);
    getHistogram("bread", 12, 13);
    getHistogram("basil_potatoes", 14, 11);
    getHistogram("pasta_with_pesto", 3, 1);
    getHistogram("rabbit", 8, 8);
    getHistogram("pasta_with_tomato_sauce", 6, 2);
    getHistogram("pilaw_rice_with_peppers_and_peas", 4, 5);
    getHistogram("pasta_with_clams_and_mussels", 12, 4);
    getHistogram("pasta_with_meat_sauce", 2, 3);
    getHistogram("seafood_salad", 3, 9);
    cout << "-> Histograms find in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

    sTime = clock();
    cout << "SVM training..." << endl;
    // Set up SVM's parameters
    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
    // Train the SVM with given parameters
    Ptr<TrainData> td = TrainData::create(inputData, ROW_SAMPLE, inputDataLables);
    svm->train(td);
    cout << "-> SVM trained in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

    sTime = clock();
    cout << "Testing images..." << endl;
    testData("grilled_pork_cutlet", 6, 6);
    testData("fish_cutlet", 10, 7);
    testData("beans", 14, 10);
    testData("salad", 19, 12);
    testData("bread", 12, 13);
    testData("basil_potatoes", 14, 11);
    testData("pasta_with_pesto", 3, 1);
    testData("rabbit", 8, 8);
    testData("pasta_with_tomato_sauce", 6, 2);
    testData("pilaw_rice_with_peppers_and_peas", 4, 5);
    testData("pasta_with_clams_and_mussels", 12, 4);
    testData("pasta_with_meat_sauce", 2, 3);
    testData("seafood_salad", 3, 9);

    cout << "-> Test completed in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

    // imwrite("sift_result.jpg", output);
    return (0);
}