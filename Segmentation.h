#ifndef segmentation_hpp
#define segmentation_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

cv::Mat segmentRectangle(const cv::Mat& image, const cv::Rect rect, const cv::Vec3b roiColor);
cv::Mat segmentImage(const cv::Mat& image, const std::vector<cv::Rect>& roisVec,
    const std::vector<int>& IDs, std::vector<int>& sortedPixels,  std::vector<cv::Mat>& predictionMasks);

double image_leftover_estimation(const cv::Mat& before, const cv::Mat& after);

std::vector<float> before_after_predicted(const std::vector<int>& before, const std::vector<int>& after);

std::vector<int> modifySortedPixels(const std::vector<int>& sortedList_before,
    const std::vector<int>& sortedList_after,
    const std::vector<int>& sortedPixels_after);

#endif /* segmentation_hpp */