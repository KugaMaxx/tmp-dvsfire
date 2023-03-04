#include "efd.hpp"
#include "dvs.hpp"

#include <math.h>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <iostream>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


efd::SelectiveDetector::SelectiveDetector(uint16_t sizeX, uint16_t sizeY, float threshold) : EventFireDetector(sizeX, sizeY) {
    _THRESH_ = threshold;
}

void efd::SelectiveDetector::findContoursRect(cv::Mat img){
    // find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // construct set of rectangle
    S = {contours.size()};
    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<cv::Point2f> centers(contours.size());
    std::vector<float> radius(contours.size());
    for(size_t i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect(contours_poly[i]);
        cv::minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
        S.push_back(boundRect[i], centers[i], radius[i]);
    }
    return;
}

inline float_t efd::SelectiveDetector::calcSimularity(int i, int j) {
    // calculate intersect
    float_t intsR = (S.rect[i] & S.rect[j]).area();
    float_t unitR = std::min(S.rect[i].area(), S.rect[j].area());
    float_t score_1 = intsR / unitR;
    
    // calculate radius
    float_t dist = cv::norm(S.center[i] - S.center[j]);
    float_t sumR = S.radius[i] + S.radius[j];
    float_t score_2 = dist < sumR ? 1. : sumR/dist;

    return 0.2 * score_1 + 0.8 * score_2;
}

std::vector<py::array_t<int32_t>> efd::SelectiveDetector::process(dvs::arrTs ts, dvs::arrX x, dvs::arrY y, dvs::arrP p) {
    cv::Mat img = efd::EventFireDetector::initialize(ts, x, y, p);
    findContoursRect(img);

    for (size_t i = 0; i < S.size; i++) {
        for (size_t j = i + 1; j < S.size; j++) {
            if (calcSimularity(i, j) >= _THRESH_) {
                S.group(i, j);
            }
        }
    }

    std::map<uint32_t, cv::Rect> rects;
    for (size_t i = 0; i < S.size; i++) {
        int ind = S.find(i);
        if (!rects.count(ind)) {
            rects[ind] = S.rect[i];
            continue;
        }
        rects[ind] |= S.rect[i];
    }

    // std::vector<py::array_t<int16_t>> result(rects.size);
    // std::vector<double> result(rects.size, 0);

    std::vector<py::array_t<int32_t>> result;
    for (auto& r : rects) {
        auto rect = r.second;
        std::vector<int32_t> vect = {rect.x, rect.y, rect.width, rect.height};
        result.push_back(py::cast(vect));
    }

    // // convert to three channels
    // std::vector<cv::Mat> channels{img, img, img};
    // cv::merge(channels, img);
            
    // cv::RNG_MT19937 rng(12345);
    // for (auto& r : rects) {
    //     cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //     cv::rectangle(img, r.second.tl(), r.second.br(), color);
    // }

    // // visualize
    // cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    // cv::imshow("Display Image", img);
    // cv::waitKey(0);

    return result;
}

PYBIND11_MODULE(eventdetector, m)
{
	py::class_<efd::SelectiveDetector>(m, "selective_detector")
		.def(py::init<uint16_t, uint16_t, float>())
		.def("process", &efd::SelectiveDetector::process);
}
