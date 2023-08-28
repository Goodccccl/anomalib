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

//void inference(const std::string enginePath, Parameters param, const cv::Mat& src, cv::Mat& result)
//{
//	std::vector<unsigned char> model_data = load_engine(enginePath);
//	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
//	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
//	if (engine == nullptr)
//	{
//		printf("Deserialize cuda engine failed!\n");
//		runtime->destroy();
//	}
//	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
//	cudaStream_t stream;
//	cudaStreamCreate(&stream);
//
//	float* inputData_vec = preprocess(src);
//	//std::ofstream infile("F:\\1.txt");
//	//for (int i = 0; i < param.batch_size * param.input_channels * param.input_height * param.input_width; i++)
//	//{
//	//	infile << inputData_vec[i] << std::endl;
//	//}
//	//infile.close();
//	void* input_mem{ nullptr };
//	cudaMalloc(&input_mem, param.batch_size * param.input_channels * param.input_height * param.input_width);
//	void* output_mem{ nullptr };
//	cudaMalloc(&output_mem, param.output_size);
//	//void* score_{ nullptr };
//	//cudaMalloc(&score_, param.score_size * sizeof(float));
//
//	CUDA_CHECK(cudaMemcpyAsync(input_mem, inputData_vec, param.batch_size * param.input_channels * param.input_height * param.input_width, cudaMemcpyHostToDevice), stream);
//
//	void* bindings[] = { input_mem, output_mem };
//	auto start = std::chrono::system_clock::now();
//	//context->enqueueV2(bindings, stream, nullptr);
//	context->enqueue(1, bindings, stream, nullptr);
//	auto end = std::chrono::system_clock::now();
//	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//	float* outputData = new float[param.output_size];
//	//float* score = new float[param.score_size];
//	//CUDA_CHECK(cudaMemcpyAsync(score, score_, param.score_size * sizeof(float), cudaMemcpyDeviceToHost), stream);
//	CUDA_CHECK(cudaMemcpyAsync(outputData, bindings[1], 1, cudaMemcpyDeviceToHost), stream);
//
//	std::ofstream outfile("F:\\1234.txt");
//	for (int i = 0; i < param.output_size; i++)
//	{
//		outfile << outputData[i] << std::endl;
//	}
//	outfile.close();
//}


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

}