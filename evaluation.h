// Evaluation functions header
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>


/* IoU

What we need for the evaluation functions:

Intersection over Union (IoU), which quantifies the degree of overlap between two regions

The value ranges from 0 to 1.
*/


float get_annotation_iou(const cv::Rect& groundTruth, const cv::Rect& prediction);
float meanaveragePrecision(const std::vector<cv::Rect>& groundTruths, const std::vector<cv::Rect>& predictions, std::vector<int>& IDs);

float get_segmentation_iou(const cv::Mat& groundTruth, const cv::Mat& prediction);
float meanintersectionoverunion(const std::vector<cv::Mat>& groundTruthMasks, const std::vector<cv::Mat>& predictionMasks, std::vector<int>& IDs);