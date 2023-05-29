#include <opencv2/opencv.hpp>
#include <iostream>

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
	return 0;
}
