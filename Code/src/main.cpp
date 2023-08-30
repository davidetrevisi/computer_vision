#include "evaluation_metrics.hpp"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    EvaluationMetrics evaluation_metrics = EvaluationMetrics();

    evaluation_metrics.players.push_back(Player(360, 424, 103, 237, 1, 0.67));
    evaluation_metrics.players.push_back(Player(445, 75, 111, 197, 1, 0.67));

    evaluation_metrics.initializePlayersGroundTruth();

    float res_1 = evaluation_metrics.meanAveragePrecision(1);
    float res_2 = evaluation_metrics.meanAveragePrecision(2);

    if (res_1 < res_2) {
        // MEAN AVERAGE PRECISION IS RES_1
    } else {
        // MEAN AVERAGE PRECISION IS RES_2
    }

    float iou_1 = evaluation_metrics.initializeSegmentationGroundTruth("im1_bin.png", 1, 2);

    return (0);
}