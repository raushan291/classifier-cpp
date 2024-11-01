#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// std::vector<std::string> classifyImages(const std::string& folderPath, cv::dnn::Net& net);
std::string classifyImage(const std::string& imagePath, cv::dnn::Net& net);
cv::Mat applySoftmax(const cv::Mat& scores);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    
    // Load the ONNX model
    std::string modelPath = argv[1];
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    if (net.empty()) {
        std::cerr << "Failed to load the model!" << std::endl;
        return -1;
    }

    // Path to the folder containing images
    // std::string folderPath = "./test-samples";

    std::string imagePath = argv[2];

    // Call the function to classify images in the folder
    // std::vector<std::string> results = classifyImages(folderPath, net);

    std::string result = classifyImage(imagePath, net);

    
    // Accumulate all results into a single string
    // std::ostringstream finalOutput;
    // for (const auto& result : results) {
    //     finalOutput << result << "\n";
    // }

    // Print the final concatenated result once
    // std::cout << finalOutput.str() << std::endl;

    std::cout << result << std::endl;

    return 0;
}


// std::vector<std::string> classifyImages(const std::string& folderPath, cv::dnn::Net& net) {
std::string classifyImage(const std::string& imagePath, cv::dnn::Net& net) {
    // std::vector<std::string> results;
    // Iterate over all images in the folder
    // for (const auto& entry : fs::directory_iterator(folderPath)) {
        // if (fs::is_regular_file(entry.path())) {
        //     std::string imagePath = entry.path().string();

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
            net.setInput(blob);

            auto start = std::chrono::high_resolution_clock::now();

            // Run forward pass
            cv::Mat output = net.forward();

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = (end - start) * 1000;

            // Apply softmax to the output
            cv::Mat probabilities = applySoftmax(output);

            // std::cout << probabilities;

            // Find the class with the highest probability
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &classIdPoint);
            int classId = classIdPoint.x;

            // Print the result
            // std::cout << "Image: " << imagePath << ", Class ID: " << classId << ", Confidence: " << confidence << ", time_taken: " << duration.count() << "ms" << std::endl;

            std::ostringstream result;
            result << "Image: " << imagePath
                << ", Class ID: " << classId
                << ", Confidence: " << confidence
                << ", time_taken: " << duration.count() << "ms";

            return result.str();

            // results.push_back(result.str());
        // }
    // }
    // return results;
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