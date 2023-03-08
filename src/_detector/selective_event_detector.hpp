#include "kore.hpp"
#include "detector.hpp"

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


namespace py = pybind11;

namespace edt {

class SelectiveDetector : public EventDetector {
public:
    float threshold;
    cv::Mat eventBinaryFrame;

    struct RegionSet {
        size_t ind  = 0;
        size_t size = 0;
        std::vector<cv::Rect>    rect;
        std::vector<cv::Point2f> center;
        std::vector<float_t>     radius;
        std::vector<uint32_t>    rank;
        std::vector<uint32_t>    label;
        
        RegionSet(){};
        RegionSet(size_t length) : size(length), ind(0) {
            rect.resize(size);
            center.resize(size);
            radius.resize(size);
            rank.resize(size);
            label.resize(size);
        }

        inline void push_back(const cv::Rect& rect_, const cv::Point2f center_, const float radius_) {
            rect[ind]   = rect_;
            center[ind] = center_;
            radius[ind] = radius_;
            label[ind]  = ind;
            ind++;
        }

        inline int find(int i) {
            return (label[i] == i) ? i : find(label[i]);
        }

        inline void group(int i, int j) {
            int x = find(i), y = find(j);
            if (x != y) {
                if (rank[x] <= rank[y]) {
                    label[x] = y;
                } else {
                    label[y] = x;
                }
                if (rank[x] == rank[y]) {
                rank[y]++;
                }
            }
            return;
        }

    } S;

    void findContoursRect(cv::Mat img) {
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
    };

    inline float_t calcSimularity(int i, int j) {
        // calculate intersect
        float_t intsR = (S.rect[i] & S.rect[j]).area();
        float_t unitR = std::min(S.rect[i].area(), S.rect[j].area());
        float_t score_1 = intsR / unitR;

        // calculate radius
        float_t dist = cv::norm(S.center[i] - S.center[j]);
        float_t sumR = S.radius[i] + S.radius[j];
        float_t score_2 = dist < sumR ? 1. : sumR/dist;

        return 0.2 * score_1 + 0.8 * score_2;
    };
};

}
