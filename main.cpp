#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <filesystem>

#include "Annotation.h"
#include "Segmentation.h"
#include "Evaluation.h"


std::vector<std::string> food_names = {
	"background",
	"pasta with pesto",
	"pasta with tomato sauce",
	"pasta with meat sauce",
	"pasta with clams and mussels",
	"pilaw rice with peppers and peas",
	"grilled pork cutlet",
	"fish cutlet",
	"rabbit",
	"seafood salad",
	"beans",
	"basil potatoes",
	"salad",
	"bread"
};


// Store all template_image and their folder names (ID numbers) in a vector
std::vector<TemplateImage> templates1;  
std::vector<TemplateImage> templates2;  
std::vector<TemplateImage> templates3;  


// method for template matching  0- basic template matching , 1- rotation and scale invariance
int method = 0 ;

int main(int argc, char* argv[])
{	
	
	if (argc != 3) {
		std::cout << "Usage: <path> <tray>" << std::endl;
		return 1;
	}
	
	// --------------- Inserting image path --------------- 
	std::cout << std::endl << "--------------- Inserting image path ---------------" << std::endl;
	std::filesystem::path path(argv[1]);
	std::cout << "Path: " << path << std::endl;
	std::filesystem::path tray(argv[2]);
	std::cout << "Tray: " << tray << std::endl << std::endl;

	std::cout << "--------------- Creating image path ---------------" << std::endl;
	std::filesystem::path difficult1Path = path / tray / "leftover1.jpg";
	std::filesystem::path difficult2Path = path / tray / "leftover2.jpg";
	std::filesystem::path difficult3Path = path / tray / "leftover3.jpg";
	std::filesystem::path beforeimagePath = path / tray / "food_image.jpg";
	std::cout << "Leftover with difficulty 1 image path: " << difficult1Path.string() << std::endl;
	std::cout << "Leftover with difficulty 2 image path: " << difficult2Path.string() << std::endl;
	std::cout << "Leftover with difficulty 3 image path: " << difficult3Path.string() << std::endl;
	std::cout << "Before image path: " << beforeimagePath.string() << std::endl << std::endl;

	std::cout << "--------------- Getting GroundTruths boxes ---------------" << std::endl;
	std::filesystem::path bbbeforeimage = path / tray / "bounding_boxes" / "food_image_bounding_box.txt";
	std::filesystem::path bbdifficulty1 = path / tray / "bounding_boxes" / "leftover1_bounding_box.txt";
	std::filesystem::path bbdifficulty2 = path / tray / "bounding_boxes" / "leftover2_bounding_box.txt";
	std::filesystem::path bbdifficulty3 = path / tray / "bounding_boxes" / "leftover3_bounding_box.txt";;
	std::cout << "Before image bounding boxes path: " << bbbeforeimage.string() << std::endl;
	std::cout << "Leftover with difficulty 1 bounding boxes path: " << bbdifficulty1.string() << std::endl;
	std::cout << "Leftover with difficulty 2 bounding boxes path: " << bbdifficulty2.string() << std::endl;
	std::cout << "Leftover with difficulty 3 bounding boxes path: " << bbdifficulty3.string() << std::endl << std::endl;

	// Setting up vector for groundtruths
	std::vector<cv::Rect> groundtruth;
	std::vector<cv::Rect> groundtruth1;
	std::vector<cv::Rect> groundtruth2;
	std::vector<cv::Rect> groundtruth3;

	std::cout << "--------------- Getting Provided Masks ---------------" << std::endl;
	std::filesystem::path mbeforeimage = path / tray / "masks" / "food_image_mask.png";
	std::filesystem::path mdifficulty1 = path / tray / "masks" / "leftover1.png";
	std::filesystem::path mdifficulty2 = path / tray / "masks" / "leftover2.png";
	std::filesystem::path mdifficulty3 = path / tray / "masks" / "leftover3.png";

	std::cout << "Before image masks path: " << mbeforeimage.string() << std::endl;
	std::cout << "Leftover with difficulty 1 masks path: " << mdifficulty1.string() << std::endl;
	std::cout << "Leftover with difficulty 2 masks path: " << mdifficulty2.string() << std::endl;
	std::cout << "Leftover with difficulty 3 masks path: " << mdifficulty3.string() << std::endl << std::endl;

	// --------------- Reading Masks images ---------------
	cv::Mat BeforeImageMask = cv::imread(mbeforeimage.string());
	cv::Mat Difficulty1ImageMask = cv::imread(mdifficulty1.string());
	cv::Mat Difficulty2ImageMask = cv::imread(mdifficulty2.string());
	cv::Mat Difficulty3ImageMask = cv::imread(mdifficulty3.string());

	// Setting up vector for provided masks
	std::vector<cv::Mat> BeforeImageprovidedMask;
	std::vector<cv::Mat> Difficulty1providedMask;
	std::vector<cv::Mat> Difficulty2providedMask;
	std::vector<cv::Mat> Difficulty3providedMask;

	// Setting up vector for segmentation masks
	std::vector<cv::Mat> BeforeImagepredMask;
	std::vector<cv::Mat> Difficulty1predMask;
	std::vector<cv::Mat> Difficulty2predMask;
	std::vector<cv::Mat> Difficulty3predMask;

	//std::cout << "--------------- Reading images ---------------" << std::endl;
	cv::Mat difficulty1 = cv::imread(difficult1Path.string());
	cv::Mat difficulty2 = cv::imread(difficult2Path.string());
	cv::Mat difficulty3 = cv::imread(difficult3Path.string());
	cv::Mat beforeimage = cv::imread(beforeimagePath.string());
	if (difficulty1.data == NULL || difficulty2.data == NULL || difficulty3.data == NULL) // safety check on cv::imread()
	{
		std::cout << "Wrong filepath";
		return 0;
	}
	
	processDirectory(path / "first_course_templates", templates1);
	processDirectory(path / "second_course_templates", templates2);
	processDirectory(path / "side_dish_templates", templates3);

	// INITIALIZE NECESSARY VARIABLES
	//for localization: matched rectangle, 
	cv::Rect bestRect;
	std::vector<cv::Rect> list_rectangles;

	// matched food name,  
	std::string food_name;
	std::vector<std::string> list_food_names;

	// template matching Max Value
	double MaxVal;
	std::vector<double> list_max_val;

	//matched food ID
	int bestTemplateID = 0;
	std::vector<int> list_best_template_IDs_beforeimage;
	std::vector<int> list_best_template_IDs_difficulty1;
	std::vector<int> list_best_template_IDs_difficulty2;
	std::vector<int> list_best_template_IDs_difficulty3;

	// for segmentation: segmented foreground pixels
	std::vector<int> foreground_pixels_beforeimage;
	std::vector<int> foreground_pixels_difficulty1;
	std::vector<int> foreground_pixels_difficulty2;
	std::vector<int> foreground_pixels_difficulty3;
	
	
	
	
	//For food localization and food segmentation you need to evaluate your system on the “before” images and 
	//the images for difficulties 1) and 2) of each provided tray in the dataset. 

	// BEFORE IMAGE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	std::cout << "--------------- Before Image Localization ---------------" << std::endl << std::endl;
	// ANNOTATION
	// --------------- First Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates1, beforeimage,method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_beforeimage.push_back(bestTemplateID);
	// --------------- Second Course Match ---------------	
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates2, beforeimage, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_beforeimage.push_back(bestTemplateID);	
	// --------------- Side dishes Match ---------------
	std::vector<std::tuple<cv::Rect, double, int>> matches = findMatchesaboveThreshold(templates3, beforeimage, method);
	for (const auto& match : matches) {
		bestRect = std::get<0>(match);
		MaxVal = std::get<1>(match);
		bestTemplateID = std::get<2>(match);
		food_name = food_names[bestTemplateID];
		list_rectangles.push_back(bestRect);
		list_food_names.push_back(food_name);
		list_max_val.push_back(MaxVal);
		list_best_template_IDs_beforeimage.push_back(bestTemplateID);
	}
	std::cout << std::endl;
	// Print Match information
	for (size_t i = 0; i < list_rectangles.size(); ++i) {

		std::cout << "Match " << i + 1 << ": Rect: (" << list_rectangles[i].x << ", " << list_rectangles[i].y << ", " << list_rectangles[i].width << ", " << list_rectangles[i].height << "), ";
		std::cout << "Name: " << list_food_names[i] << ", ID: " << list_best_template_IDs_beforeimage[i] << ", ";
		std::cout << "MaxVal: " << list_max_val[i] << std::endl;
		
		cv::Rect annotation;
		bool success = GetAnnotationByID(bbbeforeimage.string(), list_best_template_IDs_beforeimage[i], annotation);

		if (success)
		{
			groundtruth.insert(groundtruth.begin() + i, annotation);
			std::cout << "Groundtruth of ID " << list_best_template_IDs_beforeimage[i] << ": Rect=(" << groundtruth[i].x << ", " << groundtruth[i].y << ", " << groundtruth[i].width << ", " << groundtruth[i].height << ")" << std::endl << std::endl;
		}
		else
		{
			std::cout << "Failed to find annotation for ID: " << list_best_template_IDs_beforeimage[i] << std::endl << std::endl;
			annotation = cv::Rect(0, 0, 0, 0);
			groundtruth.insert(groundtruth.begin() + i, annotation);
		}
	}
	
	std::cout << "IDs found:" << std::endl;
	for (const auto& value : list_best_template_IDs_beforeimage) {
		std::cout << "ID" << value << " ";
	}

	std::cout << std::endl << std::endl << "Groundtruth vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : groundtruth) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}

	std::cout << std::endl << "Prediction vector:" << std::endl;
	// Print predicted values to check
	for (const auto& rect : list_rectangles) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl;

	// Display Matches
	cv::Mat before_img_annotated = Annotate(beforeimage, list_rectangles, list_food_names );
	show_image("Before Image Localization", before_img_annotated, 400);
	//save_image(before_img_annotated, result_path, "Before_Image_Localization");

	 //mAP
	std::cout <<  std::endl << "--------------- Evaluation mAP ---------------" << std::endl;
	float map = meanaveragePrecision(groundtruth, list_rectangles, list_best_template_IDs_beforeimage);
	std::cout << std::endl << " - mean Average Precision (Before): " << map << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;

	
	// SEGMENTATION 
	std::cout << std::endl<< "--------------- Before Image Segmentation ---------------" << std::endl << std::endl;
	cv::Mat segmentation = segmentImage(beforeimage, list_rectangles, list_best_template_IDs_beforeimage, foreground_pixels_beforeimage, BeforeImagepredMask);
	cv::Mat segmentationmask = segmentImage(BeforeImageMask, groundtruth, list_best_template_IDs_beforeimage, foreground_pixels_beforeimage, BeforeImageprovidedMask);

	std::cout << "--------------- Evaluation mIoU ---------------" << std::endl;
	float miou = meanintersectionoverunion(BeforeImageprovidedMask, BeforeImagepredMask, list_best_template_IDs_beforeimage);
	std::cout << std::endl << " - mean Intersection over Union (Before): "<< miou << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;
	show_image("Before_Image_Segmentation", segmentation, 400);
	//show_image("Segmentation of the mask of the Before image", segmentationmask, 400);
	//save_image(segmentation, result_path, "Before_Image_Segmentation");

	
	// LEFTOVER1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	std::cout << std::endl << "--------------- Leftover 1 Image Localization ---------------" << std::endl << std::endl;
	// Reassing all variables to 0
	list_rectangles = std::vector<cv::Rect>();
	list_max_val = std::vector<double>();
	list_food_names = std::vector<std::string>();
	bestRect = cv::Rect();
	MaxVal = 0.0;
	bestTemplateID = 0;
	food_name = "";
	// --------------- First Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates1, difficulty1, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty1.push_back(bestTemplateID);
	// --------------- Second Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates2, difficulty1, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty1.push_back(bestTemplateID);
	// --------------- Side dishes Match ---------------
	std::vector<std::tuple<cv::Rect, double, int>> matches1 = findMatchesaboveThreshold(templates3, difficulty1, method);
	for (const auto& match : matches1) {
		bestRect = std::get<0>(match);
		MaxVal = std::get<1>(match);
		bestTemplateID = std::get<2>(match);
		food_name = food_names[bestTemplateID];
		list_rectangles.push_back(bestRect);
		list_food_names.push_back(food_name);
		list_max_val.push_back(MaxVal);
		list_best_template_IDs_difficulty1.push_back(bestTemplateID);
	}
	std::cout << std::endl;
	std::cout << "Annotation of the before image" << std::endl;
	// Print Match information
	for (size_t i = 0; i < list_rectangles.size(); ++i) {
		std::cout << "Match " << i + 1 << ": Rect: (" << list_rectangles[i].x << ", " << list_rectangles[i].y << ", " << list_rectangles[i].width << ", " << list_rectangles[i].height << "), ";
		std::cout << "Name: " << list_food_names[i] << ", ID: " << list_best_template_IDs_difficulty1[i] << ", ";
		std::cout << "MaxVal: " << list_max_val[i] << std::endl;
		cv::Rect annotation;
		bool success = GetAnnotationByID(bbdifficulty1.string(), list_best_template_IDs_difficulty1[i], annotation);
		if (success)
		{
			groundtruth1.insert(groundtruth1.begin() + i, annotation);
			std::cout << "Groundtruth of ID: " << list_best_template_IDs_difficulty1[i] << ": Rect=(" << groundtruth1[i].x << ", " << groundtruth1[i].y << ", " << groundtruth1[i].width << ", " << groundtruth1[i].height << ")" << std::endl << std::endl;
		}
		else
		{
			std::cout << "Failed to find annotation for ID: " << list_best_template_IDs_difficulty1[i] << std::endl << std::endl;
			annotation = cv::Rect(0, 0, 0, 0);
			groundtruth1.insert(groundtruth1.begin() + i, annotation);
		}
	}
	std::cout << "IDs found:" << std::endl;
	for (const auto& value : list_best_template_IDs_difficulty1) {
		std::cout << "ID" << value << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl << "Groundtruth vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : groundtruth1) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl << "Prediction vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : list_rectangles) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl;
	// Display Matches
	cv::Mat diff1_annotated = Annotate(difficulty1, list_rectangles, list_food_names);
	show_image("Difficulty1 Localization", diff1_annotated, 400);
	//save_image(diff1_annotated, result_path, "Difficulty1 Localization");

	std::cout << std::endl << "--------------- Evaluation mAP ---------------" << std::endl;
	float map1 = meanaveragePrecision(groundtruth1, list_rectangles, list_best_template_IDs_difficulty1);
	std::cout << std::endl << " - mean Average Precision (Before): " << map1 << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;
	// --------------- Segmentation --------------- 
	std::cout << std::endl << "--------------- Leftover 1 Segmentation ---------------" << std::endl << std::endl;
	
	cv::Mat segmentation1 = segmentImage(difficulty1, list_rectangles, list_best_template_IDs_difficulty1, foreground_pixels_difficulty1, Difficulty1predMask);
	cv::Mat segmentationmask1 = segmentImage(Difficulty1ImageMask, groundtruth1, list_best_template_IDs_difficulty1, foreground_pixels_difficulty1, Difficulty1providedMask);

	std::cout << "--------------- Evaluation mIoU ---------------" << std::endl;
	float miou1 = meanintersectionoverunion(Difficulty1providedMask, Difficulty1predMask, list_best_template_IDs_difficulty1);
	std::cout << std::endl << " - mean Intersection over Union (Before): " << miou1 << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;

	// Show the binary mask
	show_image("Difficulty1_Segmentation", segmentation1, 400);
	//show_image("Segmentation of the mask of the Before image", segmentationmask1, 400);
	//save_image(segmentation1, result_path, "Difficulty1_Segmentation");
	
	// LEFTOVER2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// --------------- Leftover 2 Image Localization ---------------
	std::cout << "--------------- Leftover 2 Image Localization ---------------" << std::endl << std::endl;
	// Reassing all variables to 0
	list_rectangles = std::vector<cv::Rect>();
	list_max_val = std::vector<double>();
	list_food_names = std::vector<std::string>();
	bestRect = cv::Rect();
	MaxVal = 0.0;
	bestTemplateID = 0;
	food_name = "";
	// --------------- First Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates1, difficulty2, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty2.push_back(bestTemplateID);
	// --------------- Second Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates2, difficulty2, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty2.push_back(bestTemplateID);
	// --------------- Side dishes Match ---------------
	std::vector<std::tuple<cv::Rect, double, int>> matches2 = findMatchesaboveThreshold(templates3, difficulty2, method);
	for (const auto& match : matches2) {
		bestRect = std::get<0>(match);
		MaxVal = std::get<1>(match);
		bestTemplateID = std::get<2>(match);
		food_name = food_names[bestTemplateID];
		list_rectangles.push_back(bestRect);
		list_food_names.push_back(food_name);
		list_max_val.push_back(MaxVal);
		list_best_template_IDs_difficulty2.push_back(bestTemplateID);	}
	std::cout << std::endl;
	std::cout << "Annotation of the leftover 1 image" << std::endl;
	// Print Match information
	for (size_t i = 0; i < list_rectangles.size(); ++i) {
		std::cout << "Match " << i + 1 << ": Rect: (" << list_rectangles[i].x << ", " << list_rectangles[i].y << ", " << list_rectangles[i].width << ", " << list_rectangles[i].height << "), ";
		std::cout << "Name: " << list_food_names[i] << ", ID: " << list_best_template_IDs_difficulty2[i] << ", ";
		std::cout << "MaxVal: " << list_max_val[i] << std::endl;
		cv::Rect annotation;
		bool success = GetAnnotationByID(bbdifficulty2.string(), list_best_template_IDs_difficulty2[i], annotation);
		if (success)
		{
			groundtruth2.insert(groundtruth2.begin() + i, annotation);
			std::cout << "Groundtruth of ID: " << list_best_template_IDs_difficulty2[i] << ": Rect=(" << groundtruth2[i].x << ", " << groundtruth2[i].y << ", " << groundtruth2[i].width << ", " << groundtruth2[i].height << ")" << std::endl << std::endl;
		}
		else
		{
			std::cout << "Failed to find annotation for ID: " << list_best_template_IDs_difficulty2[i] << std::endl << std::endl;
			annotation = cv::Rect(0, 0, 0, 0);
			groundtruth2.insert(groundtruth2.begin() + i, annotation);
		}
	}
	std::cout << "IDs found:" << std::endl;
	for (const auto& value : list_best_template_IDs_difficulty2) {
		std::cout << "ID" << value << " ";
	}
	std::cout << std::endl;

	std::cout << std::endl << "Groundtruth vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : groundtruth2) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl << "Prediction vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : list_rectangles) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl;

	// Display Matches
	cv::Mat diff2_annotated = Annotate(difficulty2, list_rectangles, list_food_names);
	show_image("Difficulty2 Localization", diff2_annotated, 400);
	//save_image(diff2_annotated, result_path, "Difficulty2 Localization");

	std::cout << std::endl << "--------------- Evaluation mAP ---------------" << std::endl;
	float map2 = meanaveragePrecision(groundtruth2, list_rectangles, list_best_template_IDs_difficulty2);
	std::cout << std::endl << " - mean Average Precision (Before): " << map2 << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;
	// --------------- Segmentation --------------- 
	std::cout << "---Leftover with difficulty 2 image Segmentation---" << std::endl;
	cv::Mat segmentation2 = segmentImage(difficulty2, list_rectangles, list_best_template_IDs_difficulty2, foreground_pixels_difficulty2, Difficulty2predMask);
	cv::Mat segmentationmask2 = segmentImage(Difficulty2ImageMask, groundtruth2, list_best_template_IDs_difficulty2, foreground_pixels_difficulty2, Difficulty2providedMask);
	std::cout << "--------------- Evaluation mIoU ---------------" << std::endl;
	float miou2 = meanintersectionoverunion(Difficulty2providedMask, Difficulty2predMask, list_best_template_IDs_difficulty2);
	std::cout << std::endl << " - mean Intersection over Union (Before): " << miou2 << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;
	// Show the binary mask
	show_image("Segmentation of the difficulty2 Image result", segmentation2, 400);
	//show_image("Segmentation of the mask of the Before image", segmentationmask2, 400);
	//save_image(segmentation2, result_path, "Difficulty2_Segmentation");


	
	// LEFTOVER3 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// --------------- Leftover 3 Image Localization ---------------
	std::cout << "--------------- Leftover 3 Image Localization ---------------" << std::endl << std::endl;
	// Reassing all variables to 0
	list_rectangles = std::vector<cv::Rect>();
	list_max_val = std::vector<double>();
	list_food_names = std::vector<std::string>();
	bestRect = cv::Rect();
	MaxVal = 0.0;
	bestTemplateID = 0;
	food_name = "";	
	// --------------- First Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates1, difficulty3, method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty3.push_back(bestTemplateID);
	// --------------- Second Course Match ---------------
	std::tie(bestRect, MaxVal, bestTemplateID) = findBestMatch(templates2, difficulty3,method);
	list_rectangles.push_back(bestRect);
	food_name = food_names[bestTemplateID];
	list_food_names.push_back(food_name);
	list_max_val.push_back(MaxVal);
	list_best_template_IDs_difficulty3.push_back(bestTemplateID);
	// --------------- Side dishes Match ---------------
	std::vector<std::tuple<cv::Rect, double, int>> matches3 = findMatchesaboveThreshold(templates3, difficulty3,method);

	for (const auto& match : matches3) {
		bestRect = std::get<0>(match);
		MaxVal = std::get<1>(match);
		bestTemplateID = std::get<2>(match);
		food_name = food_names[bestTemplateID];
		list_rectangles.push_back(bestRect);
		list_food_names.push_back(food_name);
		list_max_val.push_back(MaxVal);
		list_best_template_IDs_difficulty3.push_back(bestTemplateID);
	}
	std::cout << std::endl;
	// Print Match information
	for (size_t i = 0; i < list_rectangles.size(); ++i) {
		std::cout << "Match " << i + 1 << ": Rect: (" << list_rectangles[i].x << ", " << list_rectangles[i].y << ", " << list_rectangles[i].width << ", " << list_rectangles[i].height << "), ";
		std::cout << "Name: " << list_food_names[i] << ", ID: " << list_best_template_IDs_difficulty3[i] << ", ";
		std::cout << "MaxVal: " << list_max_val[i] << std::endl;
		cv::Rect annotation;
		bool success = GetAnnotationByID(bbdifficulty3.string(), list_best_template_IDs_difficulty3[i], annotation);

		if (success)
		{
			groundtruth3.insert(groundtruth3.begin() + i, annotation);
			std::cout << "Groundtruth of ID: " << list_best_template_IDs_difficulty3[i] << ": Rect=(" << groundtruth3[i].x << ", " << groundtruth3[i].y << ", " << groundtruth3[i].width << ", " << groundtruth3[i].height << ")" << std::endl << std::endl;
		}
		else
		{
			std::cout << "Failed to find annotation for ID: " << list_best_template_IDs_difficulty3[i] << std::endl << std::endl;
			annotation = cv::Rect(0, 0, 0, 0);
			groundtruth3.insert(groundtruth3.begin() + i, annotation);
		}
	}

	std::cout << "IDs found:" << std::endl;
	for (const auto& value : list_best_template_IDs_difficulty3) {
		std::cout << "ID" << value << " ";
	}
	std::cout << std::endl;

	std::cout << std::endl << "Groundtruth vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : groundtruth3) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}

	std::cout << std::endl << "Prediction vector:" << std::endl;
	// Print groundtruth values to check
	for (const auto& rect : list_rectangles) {
		std::cout << "x: " << rect.x
			<< ", y: " << rect.y
			<< ", width: " << rect.width
			<< ", height: " << rect.height
			<< std::endl;
	}
	std::cout << std::endl;

	// Display Matches
	cv::Mat diff3_annotated = Annotate(difficulty3, list_rectangles, list_food_names);
	show_image("Difficulty3 Localization", diff3_annotated, 400);
	//save_image(diff3_annotated, result_path, "Difficulty3 Localization");

	// --------------- Segmentation --------------- 
	std::cout << "---Leftover with difficulty 3 image Segmentation---" << std::endl;

	cv::Mat segmentation3 = segmentImage(difficulty3, list_rectangles, list_best_template_IDs_difficulty3, foreground_pixels_difficulty3, Difficulty3predMask);
	cv::Mat segmentationmask3 = segmentImage(Difficulty3ImageMask, groundtruth3, list_best_template_IDs_difficulty3, foreground_pixels_difficulty3, Difficulty3providedMask);

	std::cout << "--------------- Evaluation mIoU ---------------" << std::endl;
	float miou3 = meanintersectionoverunion(Difficulty3providedMask, Difficulty3predMask, list_best_template_IDs_difficulty3);
	std::cout << std::endl << " - mean Intersection over Union (Before): " << miou3 << std::endl;
	std::cout << "-----------------------------------------------" << std::endl << std::endl;
	
	// Show the binary mask
	show_image("Segmentation of the difficulty3 Image result", segmentation3, 400);
	//show_image("Segmentation of the mask of the Before image", segmentationmask3, 400);
	//save_image(segmentation3, result_path, "Difficulty3_Segmentation");


	// --------------- Left Over Estimation --------------- 
	
	
	//For food leftover estimation, you need to evaluate your system on each pair of “before” and “after” images considering all difficulties levels.
	

	double left_over_Ri1 = image_leftover_estimation(segmentation, segmentation1);
	double left_over_Ri2 = image_leftover_estimation(segmentation, segmentation2);
	double left_over_Ri3 = image_leftover_estimation(segmentation, segmentation3);


	std::cout << std::endl;
	std::cout << "---Food Leftover Ratio Results---" << std::endl;
	std::cout << "Tray No:" << tray << std::endl;
	std::cout << "Ri: Before Image --- Difficulty1: " << left_over_Ri1 << std::endl;
	std::cout << "Ri: Before Image --- Difficulty2: " << left_over_Ri2 << std::endl;
	std::cout << "Ri: Before Image --- Difficulty3: " << left_over_Ri3 << std::endl;
	std::cout << std::endl;


	std::vector<int> modified_foreground_pixels_difficulty1 = modifySortedPixels(
		list_best_template_IDs_beforeimage,
		list_best_template_IDs_difficulty1,
		foreground_pixels_difficulty1
	);
	
	//control for Ri inputs
	//std::cout << foreground_pixels_beforeimage.size() << std::endl;
	//std::cout << modified_foreground_pixels_difficulty1.size() << std::endl;

	// Calculate quantitity of food leftover with respect to foods
	std::vector<float> Ri;
	Ri = before_after_predicted(foreground_pixels_beforeimage, modified_foreground_pixels_difficulty1);

	//print Ri s and pixels
	std::cout << std::endl;
	std::cout << "-Before Image vs Difficulty 1 ID Based Leftover Comparison-" << std::endl;
	for (int i = 0; i < list_best_template_IDs_beforeimage.size(); i++) {
		std::cout << "-Match:" << i + 1 << "-" << std::endl;
		std::cout << "ID_before_image: " << list_best_template_IDs_beforeimage[i] << " --- ";
		std::cout << "ID_difficulty1: " << list_best_template_IDs_difficulty1[i] << std::endl;
		std::cout << "Before Image Segmented Pixels: " << foreground_pixels_beforeimage[i] << std::endl;
		std::cout << "Difficulty 1 Segmented Pixels " << modified_foreground_pixels_difficulty1[i] << std::endl;
		std::cout << "Ri:" << Ri[i] << std::endl;
		std::cout << std::endl;
	}

	std::vector<int> modified_foreground_pixels_difficulty2 = modifySortedPixels(
		list_best_template_IDs_beforeimage,
		list_best_template_IDs_difficulty2,
		foreground_pixels_difficulty2
	);

	// Calculate quantitity of food leftover with respect to foods
	Ri = before_after_predicted(foreground_pixels_beforeimage, modified_foreground_pixels_difficulty2);

	//print Ri s and pixels
	std::cout << std::endl;
	std::cout << "-Before Image vs Difficulty 2 ID Based Leftover Comparison-" << std::endl;;
	for (int i = 0; i < list_best_template_IDs_beforeimage.size(); i++) {
		std::cout << "-Match:" << i + 1 << "-" << std::endl;
		std::cout << "ID_before_image: " << list_best_template_IDs_beforeimage[i] << " --- ";
		std::cout << "ID_difficulty2: " << list_best_template_IDs_difficulty2[i] << std::endl;
		std::cout << "Before Image Segmented Pixels: " << foreground_pixels_beforeimage[i] << std::endl;
		std::cout << "Difficulty 2 Segmented Pixels " << modified_foreground_pixels_difficulty2[i] << std::endl;
		std::cout << "Ri:" << Ri[i] << std::endl;
		std::cout << std::endl;
	}


	std::vector<int> modified_foreground_pixels_difficulty3 = modifySortedPixels(
		list_best_template_IDs_beforeimage,
		list_best_template_IDs_difficulty3,
		foreground_pixels_difficulty3
	);

	// Calculate quantitity of food leftover with respect to foods
	Ri = before_after_predicted(foreground_pixels_beforeimage, modified_foreground_pixels_difficulty3);

	//print Ri s and pixels
	std::cout << std::endl;
	std::cout << "-Before Image vs Difficulty 3 ID Based Leftover Comparison-" << std::endl;
	for (int i = 0; i < list_best_template_IDs_beforeimage.size(); i++) {
		std::cout << "-Match:" << i + 1 << "-" << std::endl;
		std::cout << "ID_before_image: " << list_best_template_IDs_beforeimage[i] << " --- ";
		std::cout << "ID_difficulty1: " << list_best_template_IDs_difficulty1[i] << std::endl;
		std::cout << "Before Image Segmented Pixels: " << foreground_pixels_beforeimage[i] << std::endl;
		std::cout << "Difficulty 1 Segmented Pixels " << modified_foreground_pixels_difficulty3[i] << std::endl;
		std::cout << "Ri:" << Ri[i] << std::endl;
		std::cout << std::endl;
	}

	std::cout << "FINISH" << std::endl;
	cv::waitKey(0);
	
	return 0;
}

