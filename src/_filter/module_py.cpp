#include "reclusive_event_denoisor.hpp"


namespace kpy {

class ReclusiveEventDenoisor : public edn::ReclusiveEventDenoisor {
public:
    ReclusiveEventDenoisor(int16_t sizeX_, int16_t sizeY_, std::tuple<float, float, float> params) {
        std::tie(threshold, sigmaT, sigmaS) = params;
        sizeX = ceil((float) sizeX_ / _THREAD_) * _THREAD_;
        sizeY = ceil((float) sizeY_ / _THREAD_) * _THREAD_;
        _LENGTH_ = sizeX * sizeY;
    };

    py::array_t<bool> run(const kore::EventPybind &input) {
        py::buffer_info buf = input.request();
        kore::Event *ptr = static_cast<kore::Event *> (buf.ptr);
        
        kore::EventPacket inEvent(buf.size);
        for (size_t i = 0; i < buf.size; i++) {
            inEvent[i] = ptr[i];
        }

        setCoefficient();
        Xt = (float_t *) calloc(_POLES_ * _LENGTH_, sizeof(float_t));
        Yt = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));
        Ut = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));

        std::vector<bool> vec;
        vec.reserve(inEvent.size());
        for (auto &evt : inEvent){
            uint32_t ind = evt.x() * sizeY + evt.y();
            if (evt.timestamp() - lastTimestamp < 0) {
                lastTimestamp = evt.timestamp();
            } else if (evt.timestamp() - lastTimestamp >= samplarT) {
                updateStateSpace(Xt, Yt, Ut);
                fastDericheBlur(Yt);
                lastTimestamp = evt.timestamp();
            }
            
            Ut[ind] += 1;
            if (Yt[ind] + Ut[ind] > threshold) {
                vec.push_back(true);
            } else {
                vec.push_back(false);
            }
        }

        free(Xt);
        free(Yt);
        free(Ut);

        return py::cast(vec);
    };
};

}

PYBIND11_MODULE(event_denoisor, m)
{
	py::class_<kpy::ReclusiveEventDenoisor>(m, "reclusive_event_denoisor")
		.def(py::init<int16_t, int16_t, std::tuple<float, float, float>>())
		.def("run", &kpy::ReclusiveEventDenoisor::run);
}
