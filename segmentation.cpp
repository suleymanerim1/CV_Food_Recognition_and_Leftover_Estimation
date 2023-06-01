#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

Mat segmentImage(const Mat& image, const std::vector<Rect>& rois)
{
    Mat result = image.clone();
    Mat resultLast(image.size(), CV_8UC3, Scalar(0, 0, 0));

    for (const Rect& rect : rois) {
        Mat mask, bgdModel, fgdModel;
        grabCut(image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

        Vec3b roiColor(rand() & 255, rand() & 255, rand() & 255);

        for (int i = 0; i < image.cols; i++) {
            for (int j = 0; j < image.rows; j++) {
                uchar maskValue = mask.at<uchar>(Point(i, j));
                if (maskValue == 0 || maskValue == 1 || maskValue == 2) {
                    result.at<Vec3b>(Point(i, j)) = 0;
                }
                else if (maskValue == 3) {
                    result.at<Vec3b>(Point(i, j)) = roiColor;
                }
            }
        }

        for (int i = 0; i < result.cols; i++) {
            for (int j = 0; j < result.rows; j++) {
                Vec3b pixel = result.at<Vec3b>(j, i);
                if (pixel != Vec3b(0, 0, 0)) {
                    resultLast.at<Vec3b>(j, i) = pixel;
                }
            }
        }
    }

    return resultLast;
}
