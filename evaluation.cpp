// Evaluation functions

// including header of declaration
#include "evaluation.h"

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>


// Intersection over Union (IoU) given ground truth and prediction bounding boxes
float get_iou(const cv::Rect& groundTruth, const cv::Rect& prediction)
{
    // Calculate the areas of the ground truth and prediction bounding boxes
    int groundTruthArea = groundTruth.width * groundTruth.height;
    int predictionArea = prediction.width * prediction.height;

    // Top-left and bottom-right points of the intersection rectangle:
    // 
    //keep the max between the two top-left corners
    int x1 = std::max(groundTruth.x, prediction.x);
    int y1 = std::max(groundTruth.y, prediction.y);
    //
    //keep the max between the two bottom-right corners
    int x2 = std::min(groundTruth.x + groundTruth.width, prediction.x + prediction.width);
    int y2 = std::min(groundTruth.y + groundTruth.height, prediction.y + prediction.height);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;  // check
    }

    // Calculate the intersection area 
    int intersectArea = std::max(0, x2 - x1 + 1) * std::max(0, y2 - y1 + 1);


    // Calculate the IoU by dividing the intersection area by the union of the two bounding box areas
    float iou = static_cast<float>(intersectArea) / (groundTruthArea + predictionArea - intersectArea);

    // Return the calculated IoU value
    return iou;
}

float averagePrecision(std::vector<cv::Rect>& groundTruths, std::vector<cv::Rect>& predictions)
{
    // 1) Sort Predictions, as confidence score I use IoU, I am assuming that the vectors have predictions of the initial images at the same positions
    std::vector<float> score(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        score[i] = get_iou(groundTruths[i], predictions[i]);
    }

    // Sort predictions vector based on confidence scores
    for (int i = 0; i < predictions.size() - 1; ++i)
    {
        for (int j = 0; j < predictions.size() - i - 1; ++j)
        {
            if (score[j] < score[j + 1])
            {
                cv::Rect temp = predictions[j];
                predictions[j] = predictions[j + 1];
                predictions[j + 1] = temp;
                cv::Rect temp2 = groundTruths[j];
                groundTruths[j] = groundTruths[j + 1];
                groundTruths[j + 1] = temp2;
                float temp3 = score[j];
                score[j] = score[j + 1];
                score[j + 1] = temp3;
            }
        }
    }

    for (const auto& rect : predictions) {
        std::cout << "Predictions: " << rect.x
            << ", " << rect.y
            << ", " << rect.width
            << ", " << rect.height << std::endl;
    }
    for (const auto& rect : groundTruths) {
        std::cout << "groundTruths: " << rect.x
            << ", " << rect.y
            << ", " << rect.width
            << ", " << rect.height << std::endl;
    }

    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    std::vector<float> precision(predictions.size());
    std::vector<float> recall(predictions.size());

    float ap = 0;
    float prevRecall = 0.0;
    float maxPrecision = 0.0;
    std::cout << "Predictions.size: " << predictions.size() << std::endl << std::endl;

    // PASCAL VOC 11 Point Interpolation Method for calculating the Average Precision (AP)
    for (int i = 0; i < predictions.size(); ++i)
    {
        std::cout << "i: " << i << std::endl;
        if (score[i] >= 0.5) {  // IoU threshold = 0.5
            truePositives++;
        }
        else {
            falsePositives++;
        }
        std::cout << "truepos: " << truePositives << std::endl;
        std::cout << "falsepos: " << falsePositives << std::endl;
        falseNegatives = groundTruths.size() - truePositives;
        std::cout << "falseneg: " << falseNegatives << std::endl;

        precision[i] = static_cast<float>(truePositives) / (truePositives + falsePositives);
        recall[i] = static_cast<float>(truePositives) / (truePositives + falseNegatives);
        std::cout << "Precision[i]: " << precision[i] << std::endl;
        std::cout << "Recall[i]: " << recall[i] << std::endl;




        // Calculate the maximum precision at the current recall level
        if (precision[i] > maxPrecision)
            maxPrecision = precision[i];
        std::cout << "Max Precision: " << maxPrecision << std::endl;

        // Calculate the average precision using the PASCAL VOC 11 Point Interpolation Method
        if (recall[i] != prevRecall)
        {
            ap += maxPrecision;
            prevRecall = recall[i];
        }
    }

    // Calculate the final average precision
    ap /= 11.0;

    std::cout << "Average Precision (AP): Maxprecision/11 " << ap << std::endl << std::endl << std::endl;

    return ap;
}



float mIoU(const std::vector<cv::Rect>& groundTruths, const std::vector<cv::Rect>& predictions) {
    if (groundTruths.size() != predictions.size()) {
        std::cerr << "Error: Number of ground truth objects is not equal to the number of predicted objects." << std::endl;
        return 0.0f;
    }

    float totalIoU = 0.0f;
    int numObjects = groundTruths.size();

    for (int i = 0; i < numObjects; i++) {
        float iou = get_iou(groundTruths[i], predictions[i]);
        totalIoU += iou;
    }

    float meanIoU = totalIoU / numObjects;
    return meanIoU;
}