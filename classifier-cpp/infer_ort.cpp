#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

cv::Mat applySoftmax(const cv::Mat& scores);

// void classifyImages(const std::string& folderPath, Ort::Session& session, Ort::MemoryInfo& memory_info, const std::vector<const char*>& input_node_names, const std::vector<const char*>& output_node_names, std::vector<int64_t>& input_node_dims);
std::string classifyImages(const std::string& imagePath, Ort::Session& session, Ort::MemoryInfo& memory_info, const std::vector<const char*>& input_node_names, const std::vector<const char*>& output_node_names, std::vector<int64_t>& input_node_dims);

// int main() {
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // Load the ONNX model
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // std::string model_path = "./models/vgg.onnx";
    // std::string folder_path = "./test-samples";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_node_names = {"input"};  // Adjust the input node name
    std::vector<const char*> output_node_names = {"output"};  // Adjust the output node name

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto input_node_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    
    // classifyImages(folder_path, session, memory_info, input_node_names, output_node_names, input_node_dims);
    std::string result = classifyImages(image_path, session, memory_info, input_node_names, output_node_names, input_node_dims);

    std::cout << result << std::endl;

    return 0;
}

// void classifyImages(const std::string& folderPath, Ort::Session& session, Ort::MemoryInfo& memory_info, const std::vector<const char*>& input_node_names, const std::vector<const char*>& output_node_names, std::vector<int64_t>& input_node_dims) {
std::string classifyImages(const std::string& image_path, Ort::Session& session, Ort::MemoryInfo& memory_info, const std::vector<const char*>& input_node_names, const std::vector<const char*>& output_node_names, std::vector<int64_t>& input_node_dims) {
    // for (const auto& entry : fs::directory_iterator(folderPath)) {
    //     if (fs::is_regular_file(entry.path())) {
    //         std::string image_path = entry.path().string();
            cv::Mat image = cv::imread(image_path);

            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                // continue;
                return "Error: Image loading failed.";
            }

            // Preprocess the image (resize, convert to blob, etc.)
            cv::Mat blob;
            cv::Size inputSize(224, 224);
            double scaleFactor = 1.0 / 255.0;
            cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
            cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
            image = image * scaleFactor;
            image = (image - mean) / std;
            bool swapRB = true;
            bool crop = false;

            cv::dnn::blobFromImage(image, blob, 1, inputSize, 0, swapRB, crop);

            std::vector<float> input_tensor_values(blob.begin<float>(), blob.end<float>());

            size_t input_tensor_size = input_tensor_values.size();
            std::vector<int64_t> input_tensor_shape = {1, 3, input_node_dims[2], input_node_dims[3]};

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_tensor_shape.data(), input_tensor_shape.size());
            
            auto start = std::chrono::high_resolution_clock::now();

            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            float* floatarr = output_tensors.front().GetTensorMutableData<float>();
            size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
            cv::Mat output(1, output_size, CV_32F, floatarr);

            cv::Mat probabilities = applySoftmax(output);

            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &classIdPoint);
            int classId = classIdPoint.x;

            // std::cout << "Image: " << image_path << ", Class ID: " << classId << ", Confidence: " << confidence << ", Time: " << duration.count() << "ms" << std::endl;
            
            std::ostringstream result;
            result << "Image: " << image_path
                << ", Class ID: " << classId
                << ", Confidence: " << confidence
                << ", time_taken: " << duration.count() << "ms";

            return result.str();

    //     }
    // }
}

cv::Mat applySoftmax(const cv::Mat& scores) {
    double maxVal;
    cv::minMaxLoc(scores, nullptr, &maxVal);

    cv::Mat expScores;
    cv::exp(scores - maxVal, expScores);

    cv::Scalar sumExpScores = cv::sum(expScores);

    cv::Mat probabilities = expScores / sumExpScores[0];

    return probabilities;
}
