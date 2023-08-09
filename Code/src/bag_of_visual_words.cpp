#include "bag_of_visual_words.hpp"

using namespace std;
using namespace cv;

Mat BagOfVisualWords::findHistogram(Mat descriptors)
{
    vector<DMatch> matches;
    static Ptr<BFMatcher> bfmatcher = BFMatcher::create(NORM_L2);
    bfmatcher->match(descriptors, k_centers_, matches);

    // Make a Histogram of visual words
    Mat histogram = Mat::zeros(1, dictionary_size_, CV_32F);

    for (int i = 0; i < matches.size(); i++) {
        histogram.at<float>(0, matches.at(i).trainIdx) = histogram.at<float>(0, matches.at(i).trainIdx) + 1;
    }

    return histogram;
}

BagOfVisualWords::BagOfVisualWords() {
    this->image_extension_ = ".png";
    this->dictionary_size_ = 1120; // 80 words per class more or less
    this->dataset_path_ = "../augmented_Dataset/";
}

BagOfVisualWords::BagOfVisualWords(const string& dataset_path, const string& image_extension, const int dictionary_size) {
    if (!image_extension.empty()) {
        this->image_extension_ = image_extension;
    }

    if (dictionary_size > 0) {
        this->dictionary_size_ = dictionary_size;
    }

    filesystem::path path = dataset_path;

    if (filesystem::exists(path) && filesystem::is_directory(path)) {
        this->dataset_path_ = dataset_path;
    }
}

int BagOfVisualWords::processImagesClass(const string& class_name, const int class_label)
{
    string directory_path = "";

    if (class_label < 0) {
        cout << "[ERROR] Class label is negative!" << endl;
        return -1;
    }

    if (class_name.empty()) {
        cout << "[ERROR] Class name is empty!" << endl;
        return -1;
    }
    else {
        directory_path = dataset_path_ + class_name;

        filesystem::path path = directory_path;

        if (!filesystem::exists(path) || !filesystem::is_directory(path)) {
            cout << "[ERROR] Cannot open the class folder, please check your path: " << path << endl;
            return -1;
        }
    }

    static Ptr<SIFT> sift = SIFT::create();

    for (const auto& entry : filesystem::directory_iterator(directory_path)) {
        //cout << entry.path() << endl;

        Mat image, image_grayscale;
        vector<KeyPoint> keypoints;
        Mat descriptors;

        image = imread(entry.path().string());

        if (image.data == NULL) {
            cout << "[ERROR] Cannot open the image, please check your path: " << entry.path() << endl;
            return -1;
        }

        cvtColor(image, image_grayscale, COLOR_BGR2GRAY);
        sift->detectAndCompute(image_grayscale, noArray(), keypoints, descriptors);

        descriptors_.push_back(descriptors);
        class_labels_.push_back(class_label);

        //cout << "Ciao" << endl;
    }

    return 0;
}

int BagOfVisualWords::kMeansClustering(int attempts, int iterations, const string& file_path) {
    Mat descriptors;

    for (int i = 0; i < descriptors_.size(); i++) {
        descriptors.push_back(descriptors_[i]);
    }

    cout << "Running kmeans clustering..." << endl;

    kmeans(descriptors, dictionary_size_, k_labels_, TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, iterations, 1e-4), attempts, KMEANS_PP_CENTERS, k_centers_);

    cout << "Processing done!" << endl;
    cout << "Saving to file..." << endl;

    if (!file_path.empty()) {
        FileStorage kmeans_file(file_path, FileStorage::WRITE);
        kmeans_file << "k_labels" << k_labels_ << "k_centers" << k_centers_;
        kmeans_file.release();
    }
    else {
        cout << "[ERROR] Cannot open the file, please check your path: " << file_path << endl;
        return -1;
    }


    cout << "Saving done!" << endl;

    return 0;
}

int BagOfVisualWords::importKMeans(const string& file_path) {
    cout << "Importing kmeans data from " << file_path << " ..." << endl;

    if (!file_path.empty()) {
        FileStorage kmeans_file(file_path, FileStorage::READ);
        kmeans_file["k_labels"] >> k_labels_;
        kmeans_file["k_centers"] >> k_centers_;
        kmeans_file.release();
    }
    else {
        cout << "[ERROR] Cannot open the file, please check your path: " << file_path << endl;
        return -1;
    }


    cout << "Importing done!" << endl;
    return 0;
}

int BagOfVisualWords::trainSVM(const string& file_path) {
    Mat data, data_labels;

    for (int i = 0; i < descriptors_.size(); i++) {
        Mat hist = findHistogram(descriptors_[i]);
        data.push_back(hist);
        data_labels.push_back(Mat(1, 1, CV_32SC1, class_labels_[i]));
    }

    cout << "Running SVM training..." << endl;

    svm_ = ml::SVM::create();
    svm_->setType(ml::SVM::C_SVC);
    svm_->setKernel(ml::SVM::LINEAR);
    svm_->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));

    Ptr<ml::TrainData> train_data = ml::TrainData::create(data, ml::ROW_SAMPLE, data_labels);

    svm_->train(train_data);

    cout << "Processing done!" << endl;
    cout << "Saving to file..." << endl;

    if (!file_path.empty()) {
        svm_->save(file_path);
    }

    cout << "Saving done!" << endl;
    return 0;
}

int BagOfVisualWords::importSVM(const string& file_path) {
    cout << "Importing SVM data from " << file_path << " ..." << endl;

    if (!file_path.empty()) {
        svm_ = ml::SVM::create();
        svm_ = ml::SVM::load(file_path);
    }
    else {
        cout << "[ERROR] Cannot open the file, please check your path: " << file_path << endl;
        return -1;
    }


    cout << "Importing done!" << endl;
    return 0;
}

int BagOfVisualWords::processImages() {
    cout << "Processing all the images for all the classes..." << endl;

    if (processImagesClass("grilled_pork_cutlet", 6)) {
        cout << "[ERROR] Cannot process class 'grilled_pork_cutlet'!" << endl;
        return -1;
    }

    if (processImagesClass("fish_cutlet", 7)) {
        cout << "[ERROR] Cannot process class 'fish_cutlet'!" << endl;
        return -1;
    }

    if (processImagesClass("beans", 10)) {
        cout << "[ERROR] Cannot process class 'beans'!" << endl;
        return -1;
    }

    if (processImagesClass("salad", 12)) {
        cout << "[ERROR] Cannot process class 'salad'!" << endl;
        return -1;
    }

    if (processImagesClass("bread", 13)) {
        cout << "[ERROR] Cannot process class 'bread'!" << endl;
        return -1;
    }

    if (processImagesClass("basil_potatoes", 11)) {
        cout << "[ERROR] Cannot process class 'basil_potatoes'!" << endl;
        return -1;
    }

    if (processImagesClass("pasta_with_pesto", 1)) {
        cout << "[ERROR] Cannot process class 'pasta_with_pesto'!" << endl;
        return -1;
    }

    if (processImagesClass("rabbit", 8)) {
        cout << "[ERROR] Cannot process class 'rabbit'!" << endl;
        return -1;
    }

    if (processImagesClass("pasta_with_tomato_sauce", 2)) {
        cout << "[ERROR] Cannot process class 'pasta_with_tomato_sauce'!" << endl;
        return -1;
    }

    if (processImagesClass("pilaw_rice_with_peppers_and_peas", 5)) {
        cout << "[ERROR] Cannot process class 'pilaw_rice_with_peppers_and_peas'!" << endl;
        return -1;
    }

    if (processImagesClass("pasta_with_clams_and_mussels", 4)) {
        cout << "[ERROR] Cannot process class 'pasta_with_clams_and_mussels'!" << endl;
        return -1;
    }

    if (processImagesClass("pasta_with_meat_sauce", 3)) {
        cout << "[ERROR] Cannot process class 'pasta_with_meat_sauce'!" << endl;
        return -1;
    }

    if (processImagesClass("seafood_salad", 9)) {
        cout << "[ERROR] Cannot process class 'seafood_salad'!" << endl;
        return -1;
    }

    cout << "Processing done!" << endl;
    return 0;
}

int BagOfVisualWords::predictImage(const string& file_path) {
    filesystem::path path = file_path;

    if (!filesystem::exists(path) || !filesystem::is_regular_file(path)) {
        cout << "[ERROR] Cannot open the image file, please check your path: " << path << endl;
        return -1;
    }

    static Ptr<SIFT> sift = SIFT::create();

    cout << "Predicting the image..." << endl;

    Mat image, image_grayscale;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    image = imread(file_path);

    if (image.data == NULL) {
        cout << "[ERROR] Cannot open the image, please check your path: " << file_path << endl;
        return -1;
    }

    cvtColor(image, image_grayscale, COLOR_BGR2GRAY);
    sift->detectAndCompute(image_grayscale, noArray(), keypoints, descriptors);

    Mat hist = findHistogram(descriptors);

    cout << "The image is classified as: " << svm_->predict(hist) << endl;

    return 0;
}

int BagOfVisualWords::predictImage(const cv::Mat& image) {
    if (image.data == NULL) {
        cout << "[ERROR] Cannot open the image!" << endl;
        return -1;
    }

    static Ptr<SIFT> sift = SIFT::create();

    cout << "Predicting the image..." << endl;

    Mat image_grayscale;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    cvtColor(image, image_grayscale, COLOR_BGR2GRAY);
    sift->detectAndCompute(image_grayscale, noArray(), keypoints, descriptors);

    Mat hist = findHistogram(descriptors);
    float prediction = svm_->predict(hist);

    cout << "The image is classified as: " << prediction << endl;

    return prediction;
}

int BagOfVisualWords::runFullPipeline() {
    processImages();
    kMeansClustering();
    trainSVM();

    return 0;
}