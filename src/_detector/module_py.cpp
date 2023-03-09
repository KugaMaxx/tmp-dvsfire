#include "selective_event_detector.hpp"

namespace kpy {

    class SelectiveDetector : public edt::SelectiveDetector {
    public:
        SelectiveDetector(int16_t sizeX_, int16_t sizeY_, float_t threshold_) {
            sizeX     = sizeX_;
            sizeY     = sizeY_;
            _LENGTH_  = sizeX * sizeY;
            threshold = threshold_;
        };

        std::vector<py::array_t<int32_t>> run(const kore::EventPybind &input) {
            py::buffer_info buf = input.request();
            kore::Event *ptr    = static_cast<kore::Event *>(buf.ptr);

            kore::EventPacket inEvent(buf.size);
            for (size_t i = 0; i < buf.size; i++) {
                inEvent[i] = ptr[i];
            }

            binaryImg = cv::Mat::zeros(sizeX, sizeY, CV_8UC1);
            for (auto &evt : inEvent) {
                binaryImg.at<uint8_t>(evt.x(), evt.y()) = 255;
            }

            findContoursRect(binaryImg);
            auto rankedRect = selectiveBoundingBox();

            std::vector<py::array_t<int32_t>> result;
            for (size_t i = 0; i < rankedRect.size(); i++) {
                auto rect = rankedRect[i].second;
                if (i > maxRectNum)
                    break;
                if (rect.area() < minRectArea)
                    continue;
                std::vector<int32_t> vect = {rect.x, rect.y, rect.width, rect.height};
                result.push_back(py::cast(vect));
            }

            return result;
        };
    };

}

PYBIND11_MODULE(event_detector, m) {
    py::class_<kpy::SelectiveDetector>(m, "selective_detector")
        .def(py::init<int16_t, int16_t, float>())
        .def("run", &kpy::SelectiveDetector::run);
}
