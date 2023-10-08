#include <opencv2/opencv.hpp>
#include <iostream>


cv::Mat segmentRectangle(const cv::Mat& image, const cv::Rect rect, const cv::Vec3b roiColor)
{
    cv::Mat result = image.clone();
    cv::Mat resultLast(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat mask, bgdModel, fgdModel;
    grabCut(image, mask, rect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);

    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            uchar maskValue = mask.at<uchar>(cv::Point(i, j));
            if (maskValue == 0 || maskValue == 2) {
                result.at<cv::Vec3b>(cv::Point(i, j)) = 0;
            }
            else if (maskValue == 1 || maskValue == 3) {
                result.at<cv::Vec3b>(cv::Point(i, j)) = roiColor;
            }
        }
    }

    for (int i = 0; i < result.cols; i++) {
        for (int j = 0; j < result.rows; j++) {
            cv::Vec3b pixel = result.at<cv::Vec3b>(j, i);
            if (pixel != cv::Vec3b(0, 0, 0)) {
                resultLast.at<cv::Vec3b>(j, i) = pixel;
            }
        }
    }

    return resultLast;
}

cv::Mat segmentImage(const cv::Mat& image, const std::vector<cv::Rect>& roisVec,
    const std::vector<int>& IDs, std::vector<int>& sortedPixels,  std::vector<cv::Mat>& predictionMasks)
{
    sortedPixels = {};
    cv::Mat result(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<std::tuple<int, int>> coloredPixelCounts;

    // Define a vector of colors
    std::vector<cv::Vec3b> colors = {
        cv::Vec3b(255, 255, 0),   // Cyan
        cv::Vec3b(128, 255, 0),   // Chartreuse
        cv::Vec3b(0, 255, 255),   // Yellow
        cv::Vec3b(255, 128, 0),   // Orange
        cv::Vec3b(128, 0, 255),   // Violet
        cv::Vec3b(0, 255, 128),   // Lime
        cv::Vec3b(255, 0, 255)    // Magenta
    };


    // Create segmented image
    for (size_t i = 0; i < roisVec.size(); ++i) {
        const cv::Rect& rect = roisVec[i];
        cv::Mat segmented_image;
        if (rect == cv::Rect(0, 0, 0, 0)) {
            segmented_image = cv::Mat::zeros(image.size(), image.type());
        }
        else
        { 
        segmented_image = segmentRectangle(image, rect, colors[i]);
        for (int i = 0; i < segmented_image.cols; i++) {
            for (int j = 0; j < segmented_image.rows; j++) {
                cv::Vec3b pixel = segmented_image.at<cv::Vec3b>(j, i);
                if (pixel != cv::Vec3b(0, 0, 0)) {
                    result.at<cv::Vec3b>(j, i) = pixel;
                }
            }
        }
        }
        cv::cvtColor(segmented_image, segmented_image, cv::COLOR_BGR2GRAY);
        cv::threshold(segmented_image, segmented_image, 1, 255, cv::THRESH_BINARY);

        predictionMasks.push_back(segmented_image);
    }

    std::vector<std::tuple<int, int>> sumPixels;
    // Iterate over rectangles and sum the corresponding foreground pixels
    for (size_t i = 0; i < roisVec.size(); ++i) {
        const cv::Rect& rectangle = roisVec[i];
        int pixelCount = 0;

        for (int y = rectangle.y; y < rectangle.y + rectangle.height; y++) {
            for (int x = rectangle.x; x < rectangle.x + rectangle.width; x++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(y, x);

                // Check if the pixel color matches the specified color
                if (pixel == colors[i]) {
                    pixelCount++;
                }
            }
        }
        sumPixels.push_back(std::make_tuple(pixelCount, IDs[i])); // Sum the pixel values and store the sum in the vector
    }

    // Sort the sumPixels vector based on the IDs using a custom comparator
    std::sort(sumPixels.begin(), sumPixels.end(), [](const std::tuple<int, int>& a, const std::tuple<int, int>& b) {
        return std::get<1>(a) < std::get<1>(b);
        });

    for (const auto& tuple : sumPixels) {
        int pixelCount = std::get<0>(tuple);
        sortedPixels.push_back(pixelCount); //return foreground pixels in each rectangle
    }

    return result;
}


// Calculate Ri(left over estimation ratio in pixels) for whole image before after
double image_leftover_estimation(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat before_gray, after_gray;

    // Convert the images to grayscale if they are not already
    if (before.channels() > 1) {
        cv::cvtColor(before, before_gray, cv::COLOR_BGR2GRAY);
    }
    else {
        before_gray = before.clone();
    }

    if (after.channels() > 1) {
        cv::cvtColor(after, after_gray, cv::COLOR_BGR2GRAY);
    }
    else {
        after_gray = after.clone();
    }

    // Calculate the number of non-zero pixels in the before and after images
    int numPixelsBefore = cv::countNonZero(before_gray);
    int numPixelsAfter = cv::countNonZero(after_gray);

    // Calculate the ratio of non-zero pixels in the after image to the before image
    double ratio = static_cast<double>(numPixelsAfter) / static_cast<double>(numPixelsBefore);

    return ratio;
}


//make before after image same dimension (this is necessary for next function)
std::vector<int> modifySortedPixels(const std::vector<int>& sortedList_before,
    const std::vector<int>& sortedList_after, const std::vector<int>& sortedPixels_after)
{
    std::vector<int> modifiedSortedPixels_after;
    int i = 0;
    int j = 0;

    while (i < sortedList_before.size() && j < sortedList_after.size()) {
        if (sortedList_before[i] == sortedList_after[j]) {
            modifiedSortedPixels_after.push_back(sortedPixels_after[j]);
            ++i;
            ++j;
        }
        else if (sortedList_before[i] < sortedList_after[j]) {
            modifiedSortedPixels_after.push_back(0);
            ++i;
        }
        else {
            ++j;
        }
    }

    // Add zeros for any remaining unmatched elements in sortedList_before
    while (i < sortedList_before.size()) {
        modifiedSortedPixels_after.push_back(0);
        ++i;
    }

    return modifiedSortedPixels_after;
}


// Calculates Ri's for each food in the before-after image
std::vector<float> before_after_predicted(const std::vector<int>& before, const std::vector<int>& after) {
    std::vector<float> ratio;

    if (before.size() != after.size()) {
        // Handle error: vectors must have the same size
        std::cout << "Vectors must have same size" << std::endl;
        return ratio;
    }

    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] != 0) {
            float value = static_cast<float>(after[i]) / before[i];
            ratio.push_back(value);
        }
        else {
            std::cout << "Division by Zero" << std::endl;

            // Handle error: division by zero

            return ratio;
        }
    }
    return ratio;
}