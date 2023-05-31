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