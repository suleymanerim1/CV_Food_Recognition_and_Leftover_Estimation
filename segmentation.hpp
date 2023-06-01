//
//  segmentation.hpp
//  CV_Project
//
//  Created by onuralp güvercin on 1.06.2023.
//

#ifndef segmentation_hpp
#define segmentation_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

Mat segmentImage(const Mat& image, const std::vector<Rect>& rois);

#endif /* segmentation_hpp */
