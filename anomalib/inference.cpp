#include"inference.h"
#include"cuda_utils.h"

float* preprocess(const cv::Mat& image)
{
	float* data = (float*)malloc(sizeof(float) * 3 * image.cols * image.rows);
	int i = 0;
	for (int row = 0; row < image.rows; row++) {
		uchar* uc_pixel = image.data + row * image.step;
		for (int col = 0; col < image.cols; col++) {
			data[i] = ((float)uc_pixel[2] / 255. - 0.485) / 0.229;
			data[i + image.rows * image.cols] = ((float)uc_pixel[1] / 255. - 0.456) / 0.224;
			data[i + 2 * image.rows * image.cols] = ((float)uc_pixel[0] / 255. - 0.406) / 0.225;
			uc_pixel += 3;
			++ i;
		}
	}
	return data;
}


std::vector<unsigned char> load_engine(const std::string enginePath)
{
	std::ifstream in(enginePath, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	int length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}


int volume(nvinfer1::Dims dims)
{
	int nb_dims = dims.nbDims;
	int result = 1;
	for (int i = 0; i < nb_dims; i++)
	{
		result = result * dims.d[i];
	}
	return result;
}


void inference2(const std::string enginePath, Parameters param, const cv::Mat& src, cv::Mat& result)
{
	std::vector<unsigned char> model_data = load_engine(enginePath);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
	if (engine == nullptr)
	{
		printf("Deserialize cuda engine failed!\n");
		runtime->destroy();
	}
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//std::cout << "bindings=" << engine->getNbBindings();
	int num_bindings = engine->getNbBindings();
	void* bindings[3];
	std::vector<int> bingdings_mem;
	std::vector<float*> outputs;
	for (int i = 0; i < num_bindings; i++)
	{
		const char* name;
		int mode;
		nvinfer1::DataType dtype;
		nvinfer1::Dims dims;
		int totalSize;

		mode = engine->bindingIsInput(i);
		name = engine->getBindingName(i);
		dtype = engine->getBindingDataType(i);
		dims = context->getBindingDimensions(i);
		/*std::cout << dims.d[3] << std::endl;*/
		totalSize = volume(dims) * sizeof(dtype);
		bingdings_mem.push_back(totalSize);
		cudaMalloc(&bindings[i], totalSize);
		if (!mode)
		{
			int output_size = int(totalSize / sizeof(float));
			float* output = new float[output_size];
			outputs.push_back(output);
		}
	}
	int outputs_num = outputs.size();
	float* input_vec = preprocess(src);
	cudaMemcpy(bindings[0], input_vec, bingdings_mem[0], cudaMemcpyHostToDevice);
	context->enqueueV2(bindings, stream, nullptr);
	for (int i = 0; i < outputs_num; i++)
	{
		cudaMemcpy(outputs[i], bindings[i + 1], bingdings_mem[i + 1], cudaMemcpyDeviceToHost);
	}
	
	cv::Mat anomaly_map;
	anomaly_map = cv::Mat(cv::Size(param.input_height, param.input_width), CV_32FC1, outputs[1]);
	auto hot = anomaly_map.clone();
	double minValue, maxValue;
	cv::minMaxLoc(hot, &minValue, &maxValue);
	hot = (hot - minValue) / (maxValue - minValue);
	hot.convertTo(hot, CV_8UC1, 255, 0);
	cv::imwrite("D:\\123.png", hot);
}