#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

std::mutex frameMutex;
std::queue<cv::Mat> frameQueue;
std::atomic<bool> keepRunning(true);

cv::Mat applySoftmax(const cv::Mat& scores);

bool toBool(const std::string& str) {
    return str == "true" || str == "1";
}

void captureFrames(const std::string& rtspURL) {
    cv::VideoCapture cap(rtspURL, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open RTSP stream!" << std::endl;
        keepRunning = false;
        return;
    }

    while (keepRunning) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::lock_guard<std::mutex> lock(frameMutex);
        frameQueue.push(frame);
    }

    cap.release();
}

void processFrames(cv::dnn::Net& net, bool displayFrame=false) {
    while (keepRunning) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!frameQueue.empty()) {
                frame = frameQueue.front();
                frameQueue.pop();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (displayFrame){
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(480, 270));
            cv::imshow("Video", resizedFrame);
            
            // Wait for 30 ms and break on 'q' or 'Esc' key press
            int key = cv::waitKey(30);
            if (key == 'q' || key == 27) {
                std::cout << "Playback interrupted by user." << std::endl;
                cv::destroyAllWindows();
                break;
            }
        }

        // Preprocess the frame
        cv::Mat blob;
        cv::Size inputSize(224, 224);
        double scaleFactor = 1.0 / 255.0;
        cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
        cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
        frame = frame * scaleFactor;
        frame = (frame - mean) / std;
        bool swapRB = true;
        bool crop = false;
        cv::dnn::blobFromImage(frame, blob, 1, inputSize, 0, swapRB, crop);

        // Set the input to the network
        net.setInput(blob);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat output = net.forward();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start) * 1000;

        // Apply softmax to the output
        cv::Mat probabilities = applySoftmax(output);

        // Find the class with the highest probability
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &classIdPoint);
        int classId = classIdPoint.x;

        std::cout << "Class ID: " << classId
                  << ", Confidence: " << confidence
                  << ", time_taken: " << duration.count() << "ms" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <rtsp_url>" << std::endl;
        return -1;
    }

    bool displayFrame = false;

    if (argc == 4) {
        displayFrame = toBool(argv[3]);
    }

    std::string modelPath = argv[1];
    std::string rtspURL = argv[2];

    // Load the ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Failed to load the model!" << std::endl;
        return -1;
    }

    std::thread captureThread(captureFrames, rtspURL);
    std::thread processingThread(processFrames, std::ref(net), displayFrame);

    captureThread.join();
    keepRunning = false;
    processingThread.join();

    return 0;
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
