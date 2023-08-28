#define _CRT_SECURE_NO_WARNINGS
#include"inference.h"

int main()
{
	Parameters param{};
	param.batch_size = 1;
	param.input_channels = 3;
	param.input_height = 452;
	param.input_width = 452;

	std::string img_path = "G:\\estimate_811\\membrane_cut\\Image_20230807130642587_0.bmp";
	std::string trt_path = "D:\\TensorRT-8.6.1.6\\bin\\model.trt";
	cv::Mat image = cv::imread(img_path, 1);
	//float* input_data = preprocess(image);
	cv::Mat result;
	inference2(trt_path, param, image, result);
}