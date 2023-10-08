#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <regex>
#include <fstream>
#include <iostream>
#include <map>
#include <filesystem>

#include <iostream>



void show_image(const char* name, const cv::Mat& src, const int SIZE);
void save_image(const cv::Mat& image, const std::string& savePath, const std::string& filename);

bool GetAnnotationByID(const std::string& filename, int targetID, cv::Rect& annotation);

cv::Mat Annotate(const cv::Mat& img_display, const std::vector<cv::Rect>& rectangles,const std::vector<std::string>& names);

struct TemplateImage {
    std::string ID;
    cv::Mat image;

    TemplateImage(const std::string& id, const cv::Mat& img) : ID(id), image(img) {}
};

void processDirectory(const std::filesystem::path& directoryPath, std::vector<TemplateImage>& templateImages);

struct MatchResult {
    double maxValue;
    cv::Point maxLocation;
};

MatchResult Match(const cv::Mat& img, const cv::Mat& templ, const int& method);

std::tuple<cv::Rect, double, int> findBestMatch(const std::vector<TemplateImage>& templates, const cv::Mat& img, const int& method);

std::vector<std::tuple<cv::Rect, double, int>> findMatchesaboveThreshold(const std::vector<TemplateImage>& templates, const cv::Mat& img, const int& method);