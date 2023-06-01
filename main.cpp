#include <opencv2/opencv.hpp>
#include <iostream>
# include "segmentation.hpp"

using namespace cv;
// Class Design

// After Images Struct 
struct AfterImage {
	cv::Mat image;
	int difficultyLevel;

	AfterImage(const cv::Mat& img, int level) : image(img), difficultyLevel(level) {}
};

// Tray class
class Tray {
private:
	cv::Mat beforeImage;
	std::vector<cv::Mat> afterImages;

public:
	
// Tray constructor
	Tray(const cv::Mat& before, const std::vector<cv::Mat>& after, const std::vector<int>& difficultyLevels)
		: beforeImage(before) {
		for (size_t i = 0; i < after.size(); ++i) {
			afterImages.push_back(AfterImage(after[i], difficultyLevels[i]));
		}
	}

	cv::Mat getBeforeImage() const {
		return beforeImage;
	}

	std::vector<AfterImage> getAfterImages() const {
		return afterImages;
	}

	void setBeforeImage(const cv::Mat& image) {
		beforeImage = image;
	}

	void setAfterImages(const std::vector<cv::Mat>& images, const std::vector<int>& difficultyLevels) {
		afterImages.clear();
		for (size_t i = 0; i < images.size(); ++i) {
			afterImages.push_back(AfterImage(images[i], difficultyLevels[i]));
		}
	}

	
};


int main(int argc, char** argv)
{
    std::vector<Rect> rois; // Here süleyman will give this coordinates. they are like below
    
//    rois.push_back(Rect(370, 436, 313, 331));  // Example coordinates for first ROI
//    rois.push_back(Rect(737, 145, 384, 400));  // Example coordinates for second ROI
//    rois.push_back(Rect(259, 532, 347, 357));  // Example coordinates for third ROI
//    rois.push_back(Rect(235, 79, 243, 178));   // Example coordinates for fourth ROI
// Add more ROI coordinates if needed
    
    Mat resultLast = segmentImage(image, rois);

    // Show the binary mask
    namedWindow("Segmented", WINDOW_NORMAL);
    imshow("Segmented", resultLast);
    
	return 0;
}
