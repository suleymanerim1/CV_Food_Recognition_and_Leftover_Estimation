// Evaluation functions

// including header of declaration
#include "Evaluation.h"
#include <regex>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>


    



// Intersection over Union (IoU) given ground truth and prediction bounding boxes
float get_annotation_iou(const cv::Rect& groundTruth, const cv::Rect& prediction)
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

float meanaveragePrecision(const std::vector<cv::Rect>& groundTruths, const std::vector<cv::Rect>& predictions, std::vector<int>& IDs)
{
    // Initialization of vectors
    std::vector<int> uniqueIDs(IDs.begin(), IDs.end());
    std::sort(uniqueIDs.begin(), uniqueIDs.end());
    uniqueIDs.erase(std::unique(uniqueIDs.begin(), uniqueIDs.end()), uniqueIDs.end());

   
    float sumPrecision = 0.0;
    int totalClasses = uniqueIDs.size();


    // Storing precisions per classes
    std::vector<float> classPrecisions(totalClasses, 0.0);

    // Calculate the precision for each class
    for (int i = 0; i < totalClasses; i++) {
        // Counters for true positives and total predictions for the current class
        int truePositive = 0;
        int totalPredictions = 0;
        // Iterating through predictions and their corresponding ground truth
        for (size_t j = 0; j < predictions.size(); j++) {
            const cv::Rect& prediction = predictions[j];
            const cv::Rect& groundTruth = groundTruths[j];
            int predictionID = IDs[j];
            
            // Check if the ID matches the current class.
            if (predictionID == uniqueIDs[i])
            {
                
                    // Calculate IoU
                    float iou = get_annotation_iou(groundTruth, prediction);

                    // Check if the IoU (Intersection over Union) exceeds the threshold value of 0.5 to consider the prediction as a true positive.
                    if (iou > 0.5) {
                        truePositive++;
                    }
                    totalPredictions++;
                

                
                
                 
            }
        }
            // Calculate the precision for the current class.
            float precision = static_cast<float>(truePositive) / totalPredictions;
            classPrecisions[i] = precision;  // Store the precision for the current class.
            std::cout << "Precision for ID " << uniqueIDs[i] << ": " << precision << std::endl;
            sumPrecision += precision;  // Add the precision to the total sum.
        
        
    }

    // Calculate the mean average precision (mAP).
    float meanAveragePrecision = sumPrecision / totalClasses;
    //std::cout << "mean Average Precision "  << meanAveragePrecision << std::endl;
    return meanAveragePrecision;
}

float get_segmentation_iou(const cv::Mat& groundTruth, const cv::Mat& prediction) {
    cv::Mat intersection, union_;
    cv::bitwise_and(groundTruth, prediction, intersection);
    cv::bitwise_or(groundTruth, prediction, union_);

    float intersectionArea = cv::countNonZero(intersection);
    float unionArea = cv::countNonZero(union_);

    if (unionArea == 0) {
        return 1.0;
    }

    return intersectionArea / unionArea;
}

float meanintersectionoverunion(const std::vector<cv::Mat>& groundTruthMasks, const std::vector<cv::Mat>& predictionMasks, std::vector<int>& IDs) {
    if (groundTruthMasks.size() != predictionMasks.size()) {
        std::cerr << "Error: groundTruthMasks and predictionMasks vectors must have the same size." << std::endl;
        return -1;
    }

    float sumIoU = 0.0;
    int numMasks = groundTruthMasks.size();

    for (int i = 0; i < numMasks; ++i) {
        float IoU = get_segmentation_iou(groundTruthMasks[i], predictionMasks[i]);
        std::cout << "Intersection over Union for ID " << IDs[i] << ": " << IoU << std::endl;
        sumIoU += IoU;
    }

    float mIoU = sumIoU / numMasks;
    return mIoU;
}




