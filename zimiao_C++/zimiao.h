#ifndef ZIMIAO_H
#define ZIMIAO_H
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
struct  Armor{
    cv::Point2f center;
    cv::Rect2f bbox;
    cv::Vec3f tvec;
};
struct Light{
    cv::Rect rect;
    float angle;
};
class ArmorDetector{
    public:
        ArmorDetector();
        std::vector<Armor> detect(const cv::Mat& image);
    private:
        std::vector<Light> Find_lights(const cv::Mat& binary);
        std::vector<Armor> Match_armors(const std::vector<Light>& lights);
        cv::Mat camera_matrix=(cv::Mat_<float>(3,3)<<568.18492659,0.0,270.70902258,0.0,564.52638288,238.80761855,0.0,0.0,1.0);
        cv::Mat dist_coeffs=(cv::Mat_<float>(1,5)<<-1.92828805e-01,1.19153597e+00,1.20612618e-03,-1.54875602e-02,-2.19634037e+00);
};
#endif