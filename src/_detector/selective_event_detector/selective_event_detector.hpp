#include "detector.hpp"
#include "kore.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <dv-sdk/module.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace edt {

    class SelectiveDetector : public EventDetector {
    public:
        int16_t maxRectNum;
        int16_t minRectArea;
        float_t threshold;
        cv::Mat binaryImg;

        struct RegionSet {
        private:
            size_t _ind  = 0;
            size_t _size = 0;

        public:
            std::vector<cv::Rect> rect;
            std::vector<cv::Point2f> center;
            std::vector<float_t> radius;
            std::vector<uint32_t> rank;
            std::vector<uint32_t> label;

            size_t size() {
                return _size;
            }

            RegionSet(){};
            RegionSet(size_t length) : _size(length), _ind(0) {
                rect.resize(_size);
                center.resize(_size);
                radius.resize(_size);
                rank.resize(_size);
                label.resize(_size);
            }

            inline void push_back(const cv::Rect &rect_, const cv::Point2f center_, const float radius_) {
                rect[_ind]   = rect_;
                center[_ind] = center_;
                radius[_ind] = radius_;
                label[_ind]  = _ind;
                _ind++;
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
            for (size_t i = 0; i < contours.size(); i++) {
                cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
                boundRect[i] = cv::boundingRect(contours_poly[i]);
                cv::minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
                S.push_back(boundRect[i], centers[i], radius[i]);
            }
            return;
        };

        std::vector<std::pair<int32_t, cv::Rect>> selectiveBoundingBox() {
            for (size_t i = 0; i < S.size(); i++) {
                for (size_t j = i + 1; j < S.size(); j++) {
                    if (calcSimilarity(i, j) >= threshold) {
                        S.group(i, j);
                    }
                }
            }

            std::map<int32_t, cv::Rect> rects;
            for (size_t i = 0; i < S.size(); i++) {
                int k = S.find(i);
                if (!rects.count(k)) {
                    rects[k] = S.rect[i];
                    continue;
                }
                rects[k] |= S.rect[i];
            }

            std::vector<std::pair<int32_t, cv::Rect>> rankedRect;
            for (size_t i = 0; i < rects.size(); i++) {
                if (rects[i].area() < minRectArea)
                    continue;
                rankedRect.push_back(std::make_pair(i, rects[i]));
            }
            std::sort(rankedRect.begin(), rankedRect.end(), [](auto &left, auto &right) {
                return left.second.area() > right.second.area();
            });

            return rankedRect;
        }

        inline float_t calcSimilarity(int i, int j) {
            // calculate intersect
            float_t intsR   = (S.rect[i] & S.rect[j]).area();
            float_t unitR   = std::min(S.rect[i].area(), S.rect[j].area());
            float_t score_1 = intsR / unitR;

            // calculate radius
            float_t dist    = cv::norm(S.center[i] - S.center[j]);
            float_t sumR    = S.radius[i] + S.radius[j];
            float_t score_2 = dist < sumR ? 1. : sumR / dist;

            return 0.2 * score_1 + 0.8 * score_2;
        };
    };

}
