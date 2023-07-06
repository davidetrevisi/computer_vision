#include "bag_of_visual_words.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    /*processImagesClass("grilled_pork_cutlet", 102, 6);
    processImagesClass("fish_cutlet", 90, 7);
    processImagesClass("beans", 98, 10);
    processImagesClass("salad", 114, 12);
    processImagesClass("bread", 108, 13);
    processImagesClass("basil_potatoes", 98, 11);
    processImagesClass("pasta_with_pesto", 115, 1);
    processImagesClass("rabbit", 104, 8);
    processImagesClass("pasta_with_tomato_sauce", 96, 2);
    processImagesClass("pilaw_rice_with_peppers_and_peas", 100, 5);
    processImagesClass("pasta_with_clams_and_mussels", 120, 4);
    processImagesClass("pasta_with_meat_sauce", 134, 3);
    processImagesClass("seafood_salad", 108, 9);*/

    BagOfVisualWords bovw = BagOfVisualWords();

    //bovw.processImages();
    bovw.importKMeans();
    //bovw.trainSVM();
    bovw.importSVM();
    bovw.predictImage("../Test/insalata.png");

    return (0);
}