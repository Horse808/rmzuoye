#include "zimiao.h"
#include <cmath>
const float ARMOR_WIDTH=0.225f;
const float ARMOR_HEIGHT=0.055f;
float half_w=ARMOR_WIDTH/2.0f;
float half_h=ARMOR_HEIGHT/2.0f;
ArmorDetector::ArmorDetector(){}
std::vector<cv::Point3f>obj_pts={cv::Point3f(-half_w,-half_h,0),cv::Point3f(half_w,-half_h,0),cv::Point3f(half_w,half_h,0),cv::Point3f(-half_w,half_h,0)};
//找灯条
std::vector<Light> ArmorDetector::Find_lights(const cv::Mat& binary){
    std::vector<Light> lights;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(const auto& contour:contours){
        cv::Rect rect=cv::boundingRect(contour);
        int x=rect.x,y=rect.y,w=rect.width,h=rect.height;
        float angle;
        if(h>w*0.5){
            double area=cv::contourArea(contour);
            if(area<5){
                continue;
            }
            cv::RotatedRect rotated=cv::minAreaRect(contour);
            angle=rotated.angle;
            if(w>h){
                std::swap(w,h);
                angle=90-angle;
            }
            if(std::abs(angle)>80){
                continue;
            }
        }
        Light light;
        light.rect=cv::Rect(x,y,w,h);
        light.angle=angle;
        lights.push_back(light);
    }
    return lights;
}
//匹配灯条
std::vector<Armor> ArmorDetector::Match_armors(const std::vector<Light>& lights){
    std::vector<Armor> armors;
    std::vector<bool>used(lights.size(),false);
    for(size_t i=0;i<lights.size();i++){
        if(used[i]){
            continue;
        }
        const auto& l1=lights[i];
        for(size_t j=i+1;j<lights.size();j++){
            if(used[j]){
                continue;
            }
            const auto& l2=lights[j];
            float h1=l1.rect.height,h2=l2.rect.height;
            if(std::min(h1,h2)/std::max(h1,h2)<0.7){
                continue;
            }
            const Light* left=&l1;
            const Light* right=&l2;
            if(l1.rect.x>l2.rect.x){
                std::swap(left,right);
            }
            float gap_thresh=std::max(50.0f,(left->rect.height+right->rect.height)*1.5f);
            float gap=right->rect.x-(left->rect.x+left->rect.width);
            if(gap>gap_thresh){
                continue;
            }
            if(std::abs(left->angle-right->angle)>20.0f){
                continue;
            }
            int x_min=left->rect.x;
            int x_max=right->rect.x+right->rect.width;
            int y_min=std::min(left->rect.y,right->rect.y);
            int y_max=std::max(left->rect.y+left->rect.height,right->rect.y+right->rect.height);
            cv::Rect bbox(x_min,y_min,x_max-x_min,y_max-y_min);
            cv::Point2f center((x_min+x_max)/2.0f,(y_min+y_max)/2.0f);
            cv::Point2f left_top(left->rect.x,left->rect.y),left_bottom(left->rect.x,left->rect.y+left->rect.height),
                        right_top(right->rect.x+right->rect.width,right->rect.y),right_bottom(right->rect.x+right->rect.width,right->rect.y+right->rect.height);
                        std::vector<cv::Point2f> img_pts={left_top,right_top,right_bottom,left_bottom};
            //PNP
            cv::Mat rvec,tvec;
            bool success=cv::solvePnP(obj_pts,img_pts,camera_matrix,dist_coeffs,rvec,tvec);
            if(!success){
                continue;
            }
            Armor armor;
            armor.center=center;
            armor.bbox=bbox;
            armor.tvec=cv::Vec3f(tvec.at<float>(0),tvec.at<float>(1),tvec.at<float>(2));
            armors.push_back(armor);
            used[i]=used[j]=true;
            break;            
        }
    }
    return armors;
}
std::vector<Armor> ArmorDetector::detect(const cv::Mat& image){
    cv::Mat gray;
    cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray,binary,40,255,cv::THRESH_BINARY);
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::morphologyEx(binary,binary,cv::MORPH_CLOSE,kernel);
    cv::morphologyEx(binary,binary,cv::MORPH_OPEN,kernel);
    std::vector<Light> lights=Find_lights(binary);
    std::vector<Armor> armors=Match_armors(lights);
    return armors;
}
