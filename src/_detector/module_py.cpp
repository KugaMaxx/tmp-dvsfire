#include "selective_event_detector.hpp"


namespace kpy {

class SelectiveDetector : public edt::SelectiveDetector {
public:
    SelectiveDetector(int16_t sizeX_, int16_t sizeY_, float_t threshold_) {
        sizeX = sizeX_;
        sizeY = sizeY_;
        _LENGTH_ = sizeX * sizeY;
        threshold = threshold_;
    };

    std::vector<py::array_t<int32_t>> run(const kore::EventPybind &input) {
        py::buffer_info buf = input.request();
        kore::Event *ptr = static_cast<kore::Event *> (buf.ptr);
        
        kore::EventPacket inEvent(buf.size);
        for (size_t i = 0; i < buf.size; i++) {
            inEvent[i] = ptr[i];
        }

        eventBinaryFrame = cv::Mat::zeros(sizeX, sizeY, CV_8UC1);
        for (auto &evt : inEvent) {
            eventBinaryFrame.at<cv::Vec2b>(evt.x(), evt.y()) = 255;
        }
        
        findContoursRect(eventBinaryFrame);
        for (size_t i = 0; i < S.size; i++) {
            for (size_t j = i + 1; j < S.size; j++) {
                if (calcSimularity(i, j) >= threshold) {
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

        std::vector<py::array_t<int32_t>> result;
        for (auto& r : rects) {
            auto rect = r.second;
            std::vector<int32_t> vect = {rect.x, rect.y, rect.width, rect.height};
            result.push_back(py::cast(vect));
        }

        // convert to three channels
        auto img = eventBinaryFrame;
        std::vector<cv::Mat> channels{img, img, img};
        cv::merge(channels, img);
                
        // cv::RNG_MT19937 rng(12345);
        // for (auto& r : rects) {
        //     cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        //     cv::rectangle(img, r.second.tl(), r.second.br(), color);
        // }
        // cv::imwrite("test.jpg", img); 
        // visualize
        cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
        cv::imshow("Display Image", img);
        cv::waitKey(0);

        return result;
    };
};

}

PYBIND11_MODULE(event_detector, m)
{
	py::class_<kpy::SelectiveDetector>(m, "selective_detector")
		.def(py::init<int16_t, int16_t, float>())
		.def("run", &kpy::SelectiveDetector::run);
}
