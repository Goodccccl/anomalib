#pragma once
#include<stdio.h>
#include<opencv.hpp>
#include"logger.h"
#include<NvInfer.h>
#include<fstream>
#include<cuda.h>
#include"parameters.h"

float* preprocess(const cv::Mat& image);



//void inference(const std::string enginePath, Parameters param, const cv::Mat& src, cv::Mat& result);

void inference2(const std::string enginePath, Parameters param, const cv::Mat& src, cv::Mat& result);