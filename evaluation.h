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

float get_iou(const cv::Rect& groundTruth, const cv::Rect& prediction);
float mIoU(const std::vector<cv::Rect>& groundTruths, const std::vector<cv::Rect>& predictions);

float averagePrecision(std::vector<cv::Rect>& groundTruths, std::vector<cv::Rect>& predictions);