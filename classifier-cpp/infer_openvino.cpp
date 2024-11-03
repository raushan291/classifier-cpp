#include <iostream>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace fs = std::filesystem;
using namespace ov; // Using the OpenVINO namespace

// void classifyImages(const std::string& folderPath, const std::string& modelPath, const std::string& device);
std::string classifyImages(const std::string& imagePath, const std::string& modelPath, const std::string& device);
cv::Mat applySoftmax(const cv::Mat& scores);

// int main() {
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // Load the OpenVINO model
    std::string modelPath = argv[1];
    std::string device = "CPU"; // Specify the target device

    // Path to the folder containing images
    // std::string folderPath = "./test-samples";

    std::string imagePath = argv[2];

    // Call the function to classify images in the folder
    // classifyImages(folderPath, modelPath, device);
    std::string result = classifyImages(imagePath, modelPath, device);

    std::cout << result << std::endl;


    return 0;
}

// void classifyImages(const std::string& folderPath, const std::string& modelPath, const std::string& device) {
std::string classifyImages(const std::string& imagePath, const std::string& modelPath, const std::string& device) {
    // Initialize the OpenVINO inference engine
    Core ie;

    // Load the network model
    auto model = ie.read_model(modelPath);
    
    // Compile the model for the specified device
    auto compiledModel = ie.compile_model(model, device);
    
    // Create an inference request
    InferRequest inferRequest = compiledModel.create_infer_request();

    // Get input and output information
    const auto& inputShapes = model->inputs();
    const auto& outputShapes = model->outputs();

    // Iterate over all images in the folder
    // for (const auto& entry : fs::directory_iterator(folderPath)) {
    //     if (fs::is_regular_file(entry.path())) {
    //         std::string imagePath = entry.path().string();

            // Load and preprocess the image
            cv::Mat image = cv::imread(imagePath);

            if (image.empty()) {
                std::cerr << "Failed to load image: " << imagePath << std::endl;
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

            // Set the input to the network
            auto inputBlob = inferRequest.get_input_tensor();
            std::memcpy(inputBlob.data(), blob.data, blob.total() * blob.elemSize());

            auto start = std::chrono::high_resolution_clock::now();

            // Run forward pass
            inferRequest.infer();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = (end - start) * 1000;

            // Get the output
            auto outputBlob = inferRequest.get_output_tensor();
            auto outputBlobData = outputBlob.data<float>();

            // Assuming the output is a vector of probabilities
            cv::Mat outputMat(1, outputBlob.get_shape()[1], CV_32F, outputBlobData);

            // Apply softmax to the output
            cv::Mat probabilities = applySoftmax(outputMat);

            // Find the class with the highest probability
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &classIdPoint);
            int classId = classIdPoint.x;

            // Print the result
            // std::cout << "Image: " << imagePath << ", Class ID: " << classId << ", Confidence: " << confidence << ", time taken: " << duration.count() << "ms" << std::endl;

            std::ostringstream result;
            result << "Image: " << imagePath
                << ", Class ID: " << classId
                << ", Confidence: " << confidence
                << ", time_taken: " << duration.count() << "ms";

            return result.str();
    //     }
    // }
}

cv::Mat applySoftmax(const cv::Mat& scores) {
    // Find the maximum score for numerical stability
    double maxVal;
    cv::minMaxLoc(scores, nullptr, &maxVal);

    // Subtract maxVal for numerical stability and then exponentiate
    cv::Mat expScores;
    cv::exp(scores - maxVal, expScores);

    // Sum the exponentiated scores
    cv::Scalar sumExpScores = cv::sum(expScores);

    // Normalize to get the probabilities
    cv::Mat probabilities = expScores / sumExpScores[0];

    return probabilities;
}
