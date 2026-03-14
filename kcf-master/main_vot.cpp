#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "kcf.h"

// Protocol commands
enum Command {
    INIT,
    TRACK,
    QUIT
};

// Helper function to parse command from stdin
Command parseCommand(const std::string& line) {
    if (line.substr(0, 4) == "INIT") return INIT;
    if (line.substr(0, 5) == "TRACK") return TRACK;
    if (line.substr(0, 4) == "QUIT") return QUIT;
    return QUIT; // Default to quit on unknown command
}

// Helper function to parse bounding box from string
cv::Rect parseBoundingBox(const std::string& bbox_str) {
    std::istringstream iss(bbox_str);
    double x, y, w, h;
    iss >> x >> y >> w >> h;
    return cv::Rect(x, y, w, h);
}

// Helper function to output bounding box
void outputBoundingBox(const cv::Rect& rect, const double& confidence = 1.0) {
    std::cout << "BBOX " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << " " << confidence << std::endl;
    std::cout.flush();
}

// Helper function to output status
void outputStatus(const std::string& status) {
    std::cout << "STATUS " << status << std::endl;
    std::cout.flush();
}

int main()
{
    KCF_Tracker tracker;
    cv::Mat image;
    bool initialized = false;
    
    std::string line;
    BBox_c bb;
    double avg_time = 0.;
    int frames = 0;
    
    outputStatus("READY");
    
    // Main command loop
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        Command cmd = parseCommand(line);
        
        switch (cmd) {
            case INIT: {
                // Expected format: INIT <image_path> <x> <y> <width> <height>
                std::istringstream iss(line);
                std::string cmd_str, image_path;
                double x, y, w, h;
                
                if (iss >> cmd_str >> image_path >> x >> y >> w >> h) {
                    image = cv::imread(image_path, cv::IMREAD_COLOR);
                    if (!image.empty()) {
                        cv::Rect init_rect(x, y, w, h);
                        tracker.init(image, init_rect);
                        initialized = true;
                        outputBoundingBox(init_rect);
                        outputStatus("INITIALIZED");
                    } else {
                        outputStatus("ERROR: Could not load image");
                    }
                } else {
                    outputStatus("ERROR: Invalid INIT command format");
                }
                break;
            }
            
            case TRACK: {
                if (!initialized) {
                    outputStatus("ERROR: Tracker not initialized");
                    break;
                }
                
                // Expected format: TRACK <image_path>
                std::istringstream iss(line);
                std::string cmd_str, image_path;
                
                if (iss >> cmd_str >> image_path) {
                    image = cv::imread(image_path, cv::IMREAD_COLOR);
                    if (!image.empty()) {
                        double time_profile_counter = cv::getCPUTickCount();
                        tracker.track(image);
                        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
                        
                        avg_time += time_profile_counter/((double)cv::getTickFrequency()*1000);
                        frames++;
                        
                        bb = tracker.getBBox();
                        cv::Rect result_rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h);
                        
                        outputBoundingBox(result_rect, tracker.getMaxResponse());
                        outputStatus("TRACKED");
                    } else {
                        outputStatus("ERROR: Could not load image");
                    }
                } else {
                    outputStatus("ERROR: Invalid TRACK command format");
                }
                break;
            }
            
            case QUIT:
                if (frames > 0) {
                    std::cerr << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;
                }
                outputStatus("QUIT");
                return EXIT_SUCCESS;
        }
    }
    
    return EXIT_SUCCESS;
}