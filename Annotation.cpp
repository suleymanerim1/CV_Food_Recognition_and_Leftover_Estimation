#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <regex>
#include <fstream>
#include <iostream>
#include <map>
#include <filesystem>
#include "Annotation.h"

void show_image(const char* name, const cv::Mat& src, const int SIZE) {
    namedWindow(name, cv::WINDOW_NORMAL);
    cv::resizeWindow(name, SIZE, SIZE);
    cv::imshow(name, src);
}

void save_image(const cv::Mat& image, const std::string& savePath, const std::string& filename)
{
    if (image.empty())
    {
        std::cout << "Invalid image data." << std::endl;
    }

    std::string fullPath = savePath +"/"+ filename + ".jpg";
    //std::cout << "path:" << fullPath << std::endl;

    if (!cv::imwrite(fullPath, image))
    {
        std::cout << "Failed to save the image: " << filename << std::endl;
    }
}


bool GetAnnotationByID(const std::string& filename, int targetID, cv::Rect& annotation)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::regex idRegex("ID: (\\d+); \\[(\\d+), (\\d+), (\\d+), (\\d+)\\]");
        std::smatch match;
        if (std::regex_search(line, match, idRegex))
        {
            int id = std::stoi(match[1]);

            if (id == targetID)
            {
                int x = std::stoi(match[2]);
                int y = std::stoi(match[3]);
                int width = std::stoi(match[4]);
                int height = std::stoi(match[5]);

                annotation = cv::Rect(x, y, width, height);

                file.close();
                return true;
            }
        }
    }

    file.close();
    return false;
}

void processDirectory(const std::filesystem::path& directoryPath, std::vector<TemplateImage>& templateImages) {
    // Iterate through the directory
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_directory()) {
            // Get the current folder name (ID number)
            std::string ID = entry.path().filename().string();

            // Recursive call to process subdirectories
            processDirectory(entry.path(), templateImages);

            // Skip adding the folder name to the templateImages vector
            continue;
        }

        if (entry.is_regular_file()) {
            // Load the image file
            std::string imagePath = entry.path().string();
            //std::cout << "ID: " << directoryPath.filename().string() << ", Image: " << imagePath << std::endl;
            cv::Mat image = cv::imread(imagePath);

            // Create a TemplateImage instance and add it to the templateImages vector
            TemplateImage templateImage(directoryPath.filename().string(), image);
            templateImages.push_back(templateImage);
        }
    }
}

cv::Mat Annotate(const cv::Mat& img_display, const std::vector<cv::Rect>& rectangles,const std::vector<std::string>& names) {
    cv::Mat img_result = img_display.clone();

    // Define a vector of colors for rectangles and text labels
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 255, 0),   // Cyan
        cv::Scalar(128, 255, 0),   // Chartreuse
        cv::Scalar(0, 255, 255),   // Yellow
        cv::Scalar(255, 128, 0),   // Orange
        cv::Scalar(128, 0, 255),   // Violet
        cv::Scalar(0, 255, 128),   // Lime
        cv::Scalar(255, 0, 255)    // Magenta
    };

    int textPadding = 10;  // Padding between each name
    int textOffset = 30;   // Vertical offset of the names from the top

    // Draw rectangles with different colors
    for (size_t i = 0; i < rectangles.size(); i++) {
        cv::Scalar color = colors[i % colors.size()];  // Cycle through the colors

        cv::rectangle(img_result, rectangles[i], color, 2, cv::LINE_8);
    }

    // Write food names in the top-left corner of the image
    for (size_t i = 0; i < names.size(); i++) {
        cv::Scalar color = colors[i % colors.size()];  // Cycle through the colors

        // Calculate the position for the text label
        int textX = textPadding;
        int textY = (i + 1) * (textOffset + textPadding) + textPadding;

        // Draw a filled rectangle as the background of the text label
        cv::rectangle(img_result, cv::Point(textX, textY - textPadding),
            cv::Point(textX + textPadding + 150, textY + 10),
            color, cv::FILLED);

        // Write the food name as the text label
        cv::putText(img_result, names[i], cv::Point(textX + textPadding, textY),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    return img_result;
}

MatchResult Match(const cv::Mat& img, const cv::Mat& templ, const int& method)
{   
    //method == 0 for basic template matching
    //method == 1 for rotation and scale invariance
    std::vector<double> scales;
    std::vector<double> rotations;

    if (method == 0)
    {
        scales = { 1.0 };
        rotations = { 0 };
    }
    if (method == 1) 
    {
        scales = { 0.4, 0.6, 0.8, 1.0 };  // Example scales, modify as needed
        rotations = { 0, 30, 60, 90, 120, 180 };  // Example rotation angles, modify as needed
    }


    double maxVal = -std::numeric_limits<double>::infinity();
    cv::Point maxLoc;
    double curr_Scale = 0.0;
    double curr_Rotation = 0.0;
    cv::Mat transformed_img;
    double currentMaxVal;
    cv::Point currentMaxLoc;

    for (double scale : scales) {
        int scaledWidth = static_cast<int>(img.cols * scale);
        int scaledHeight = static_cast<int>(img.rows * scale);

        if (scaledWidth < templ.cols || scaledHeight < templ.rows) {
            //std::cout << "Image size is smaller then template size. Scale: " << scale << std::endl;
            continue;  // Skip the scale if the image size is smaller than the template size
        }

        cv::resize(img, transformed_img, cv::Size(scaledWidth, scaledHeight));

        for (double rotation : rotations) {
            cv::Mat rotatedTempl;
            cv::Point center(templ.cols / 2, templ.rows / 2);
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotation, 1.0);
            cv::warpAffine(templ, rotatedTempl, rotationMatrix, templ.size());

            int result_cols = transformed_img.cols - rotatedTempl.cols + 1;
            int result_rows = transformed_img.rows - rotatedTempl.rows + 1;
            cv::Mat result(result_rows, result_cols, CV_32FC1);

            //std::cout << "Try rotation: " << rotation << std::endl;
            cv::matchTemplate(transformed_img, rotatedTempl, result, cv::TM_CCOEFF_NORMED);

            cv::minMaxLoc(result, nullptr, &currentMaxVal, nullptr, &currentMaxLoc);

            if (currentMaxVal > maxVal) {
                maxVal = currentMaxVal;
                maxLoc = cv::Point(currentMaxLoc.x / scale, currentMaxLoc.y / scale);
                curr_Scale = scale;
                curr_Rotation = rotation;
            }
        }
    }

    //std::cout << "Maximum value: " << maxVal << std::endl;
    //std::cout << "Location: (" << maxLoc.x << ", " << maxLoc.y << ")" << std::endl;
    //std::cout << "Scale: " << curr_Scale << std::endl;
    //std::cout << "Rotation: " << curr_Rotation << std::endl;


    return { maxVal, maxLoc };
}

std::tuple<cv::Rect, double, int> findBestMatch(const std::vector<TemplateImage>& templates, const cv::Mat& img, const int& method) {
    double highestMaxVal = -std::numeric_limits<double>::infinity();
    cv::Rect bestRect;
    int bestTemplateID;

    for (int i = 0; i < templates.size(); i++) {
        const TemplateImage& templ = templates[i];

        MatchResult matchResult = Match(img, templ.image, method);

        if (matchResult.maxValue > highestMaxVal) {
            highestMaxVal = matchResult.maxValue;
            bestRect = cv::Rect(matchResult.maxLocation, templ.image.size());
            bestTemplateID = std::stoi(templ.ID); //cast from str to int
        }
    }

    return std::make_tuple(bestRect, highestMaxVal, bestTemplateID);
}



std::vector<std::tuple<cv::Rect, double, int>> findMatchesaboveThreshold(const std::vector<TemplateImage>& templates, const cv::Mat& img, const int& method)
{
    std::map<int, std::tuple<cv::Rect, double, int>> bestMatches; // Store the best match for each ID

    std::map<int, double> thresholds = {
        // ID vs threshold
        { 1, 0.95 }, // pesto
        { 2, 0.95 }, // tomato
        { 3, 0.95 }, // meat
        { 4, 0.99 }, // clams and mussels
        { 5, 0.95 }, // rice
        { 6, 0.95 }, // pork
        { 7, 0.95 }, // salad
        { 8, 0.99 },// bread
        { 9, 0.95 }, // beans
        { 10, 0.95 }, // beans
        { 11, 0.95 }, // potatoes
        { 12, 0.95 }, // salad
        { 13, 0.99 } // bread
    };

    for (int i = 0; i < templates.size(); i++) {
        const TemplateImage& templ = templates[i];

        MatchResult matchResult = Match(img, templ.image, method);

        double threshold = thresholds[std::stoi(templ.ID)]; // Get the threshold for the current template ID

        if (matchResult.maxValue > threshold) {
            cv::Rect rect = cv::Rect(matchResult.maxLocation, templ.image.size());
            int templateID = std::stoi(templ.ID);

            // Check if a match already exists for the current ID
            if (bestMatches.find(templateID) == bestMatches.end() || matchResult.maxValue < std::get<1>(bestMatches[templateID])) {
                bestMatches[templateID] = std::make_tuple(rect, matchResult.maxValue, templateID);
            }
        }
    }

    // Convert the map of best matches to a vector and return
    std::vector<std::tuple<cv::Rect, double, int>> matches;
    for (const auto& match : bestMatches) {
        matches.push_back(match.second);
    }

    return matches;
}