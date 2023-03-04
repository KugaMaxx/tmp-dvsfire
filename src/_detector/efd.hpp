#ifndef EFD_H
#define EFD_H

#include "dvs.hpp"

#include <vector>
#include <stdlib.h>
#include <iostream>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>


namespace py = pybind11;

namespace efd {
    class EventFireDetector {
    protected:
        int32_t sizeX;
        int32_t sizeY;
        uint32_t evlen;  // Length of noise events
        
        bool *ptrp;
        uint16_t *ptrx;
        uint16_t *ptry;
        uint64_t *ptrts;

        cv::Mat pMat;
        cv::Mat nMat;

    public:
        EventFireDetector(uint16_t sizeX, uint16_t sizeY) : sizeX(sizeX), sizeY(sizeY) {};
        virtual ~EventFireDetector() {};
        cv::Mat initialize(dvs::arrTs ts, dvs::arrX x, dvs::arrY y, dvs::arrP p) {
            py::buffer_info bufp = p.request(), bufx = x.request(), bufy = y.request(), bufts = ts.request();
	        assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);
            
            evlen = bufts.size;
            ptrts = static_cast<uint64_t *> (bufts.ptr);
            ptrx  = static_cast<uint16_t *> (bufx.ptr);
            ptry  = static_cast<uint16_t *> (bufy.ptr);
            ptrp  = static_cast<bool *>     (bufp.ptr);

            cv::Mat binImg = cv::Mat::zeros(sizeX, sizeY, CV_8UC1);
            pMat = cv::Mat::zeros(sizeX, sizeY, CV_8UC1);
            nMat = cv::Mat::zeros(sizeX, sizeY, CV_8UC1);

            for (size_t i = 0; i < evlen; i++) {
                dvs::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
                pMat.at<uint8_t>(event.x, event.y) += (event.p == 1);
                nMat.at<uint8_t>(event.x, event.y) += (event.p != 1);
                binImg.at<uint8_t>(event.x, event.y) = 255;
            }

	        return binImg;
        }
    };

    class SelectiveDetector : public EventFireDetector {
    private:
        float _THRESH_;
        float _DIAGONAL_;

        typedef struct RegionSet {
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

            inline void push_back(const cv::Rect& rect, const cv::Point2f center, const float radius) {
                this->rect[ind] = rect;
                this->center[ind] = center;
                this->radius[ind] = radius;
                this->label[ind] = ind;
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

        } RegionSet;

        RegionSet S;

        void findContoursRect(cv::Mat img);
        inline float_t calcSimularity(int i, int j);

    public:
        SelectiveDetector(uint16_t sizeX, uint16_t sizeY, float threshold);
        std::vector<py::array_t<int32_t>> process(dvs::arrTs ts, dvs::arrX x, dvs::arrY y, dvs::arrP p);
    };
}

#endif
