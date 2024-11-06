#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <vector>
#include <map>

std::mutex queueMutex;
std::map<int, std::queue<cv::Mat>> frameQueues;
std::atomic<bool> keepRunning(true);

cv::Mat applySoftmax(const cv::Mat& scores);

bool toBool(const std::string& str) {
    return str == "true" || str == "1";
}

void captureFrames(const std::string& rtspURL, int camId) {
    cv::VideoCapture cap(rtspURL, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open RTSP stream for camera " << camId << "!" << std::endl;
        keepRunning = false;
        return;
    }

    while (keepRunning) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::lock_guard<std::mutex> lock(queueMutex);
        frameQueues[camId].push(frame);
    }

    cap.release();
}

void processFrames(cv::dnn::Net& net, int camId, bool displayFrame=false) {
    while (keepRunning) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (!frameQueues[camId].empty()) {
                frame = frameQueues[camId].front();
                frameQueues[camId].pop();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (displayFrame) {
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(480, 270));
            cv::imshow("Camera " + std::to_string(camId), resizedFrame);

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

        net.setInput(blob);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat output = net.forward();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start) * 1000;

        cv::Mat probabilities = applySoftmax(output);
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &classIdPoint);
        int classId = classIdPoint.x;

        std::cout << "Camera ID: " << camId
                  << ", Class ID: " << classId
                  << ", Confidence: " << confidence
                  << ", time_taken: " << duration.count() << "ms" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <display_frame> <rtsp_url1> [<rtsp_url2> ... <rtsp_urlN>]" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    bool displayFrame = toBool(argv[2]);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Failed to load the model!" << std::endl;
        return -1;
    }

    std::vector<std::thread> captureThreads;
    std::vector<std::thread> processingThreads;

    for (int i = 3; i < argc; ++i) {
        std::string rtspURL = argv[i];
        int camId = i - 3;

        frameQueues[camId] = std::queue<cv::Mat>();  // Initialize frame queue for each camera

        captureThreads.emplace_back(captureFrames, rtspURL, camId);
        processingThreads.emplace_back(processFrames, std::ref(net), camId, displayFrame);
    }

    for (auto& thread : captureThreads) thread.join();
    keepRunning = false;
    for (auto& thread : processingThreads) thread.join();

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
