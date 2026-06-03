#include "zimiao.h"
#include<iostream>
#include <opencv2/opencv.hpp>
int main(int argc,char** argv){
    cv::VideoCapture cap;
    if(argc>1){
        cap.open(argv[1]);
    }
    else{
        cap.open(0);
    }
    if(!cap.isOpened()){
        std::cerr<<"error"<<std::endl;
        return -1;
    }
    ArmorDetector detector;
    cv::Mat frame;
    while (true){
        cap>>frame;
        if(frame.empty()){
            break;
        }
        auto armors=detector.detect(frame);
        for(const auto& armor:armors){
            cv::rectangle(frame,armor.bbox,cv::Scalar(0,255,0),2);
            cv::circle(frame,armor.center,5,cv::Scalar(0,0,255),-1);
            char text[32];
            sprintf(text,"%.1fm",armor.tvec[2]);
            cv::putText(frame,text,cv::Point(armor.bbox.x,armor.bbox.y-5),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,0),1);
            cv::imshow("Armor",frame);
            if(cv::waitKey(30)=='q'){
                break;
            }
        }
    }
    return 0;
}