#include "reclusive_event_denoisor.hpp"

namespace kdv {

    class ReclusiveEventDenoisor : public edn::ReclusiveEventDenoisor, public dv::ModuleBase {
    public:
        static const char *initDescription() {
            return "Reclusive filter using state space and deriche blur.";
        };

        static void initInputs(dv::InputDefinitionList &in) {
            in.addEventInput("events");
        };

        static void initOutputs(dv::OutputDefinitionList &out) {
            out.addEventOutput("events");
        };

        static void initConfigOptions(dv::RuntimeConfig &config) {
            config.add("samplarT", dv::ConfigOption::floatOption("Log scale to slice event stream.", -0.3, -1.0, 2.0));
            config.add("sigmaS", dv::ConfigOption::floatOption("Spatial blur coefficient.", 1.0, 0.1, 3.0));
            config.add("sigmaT", dv::ConfigOption::intOption("Time sigma.", 1, 1, 5));
            config.add("threshold", dv::ConfigOption::floatOption("Threshold value.", 0.5, -1.0, 3.0));

            config.setPriorityOptions({"samplarT", "sigmaT", "sigmaS", "threshold"});
        };

        ReclusiveEventDenoisor() {
            sizeX    = ceil((float)inputs.getEventInput("events").sizeX() / _THREAD_) * _THREAD_;
            sizeY    = ceil((float)inputs.getEventInput("events").sizeY() / _THREAD_) * _THREAD_;
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));

            Xt = (float_t *)calloc(_POLES_ * _LENGTH_, sizeof(float_t));
            Yt = (float_t *)calloc(1 * _LENGTH_, sizeof(float_t));
            Ut = (float_t *)calloc(1 * _LENGTH_, sizeof(float_t));
        };

        ~ReclusiveEventDenoisor() {
            free(Xt);
            free(Yt);
            free(Ut);
        }

        void run() override {
            auto inEvent  = inputs.getEventInput("events").events();
            auto outEvent = outputs.getEventOutput("events").events();

            if (!inEvent) {
                return;
            }

            for (auto &evt : inEvent) {
                bool isNoise = calculateDensity(evt.x(), evt.y(), evt.timestamp());

                if (isNoise) {
                    outEvent << evt;
                }
            }
            outEvent << dv::commit;
        };

        void configUpdate() override {
            samplarT  = config.getFloat("samplarT");
            sigmaS    = config.getFloat("sigmaS");
            sigmaT    = config.getInt("sigmaT");
            threshold = config.getFloat("threshold");
            setCoefficient();
        };
    };

}

registerModuleClass(kdv::ReclusiveEventDenoisor)
