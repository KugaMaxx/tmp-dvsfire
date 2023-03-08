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
        config.add("sigmaT",    dv::ConfigOption::floatOption("Time sigma (* 0.1/ms).",     1.0, 0.001, 10.0));
        config.add("sigmaS",    dv::ConfigOption::floatOption("Spatial blur coefficient.",  1.0, 0.1, 10.0));
        config.add("threshold", dv::ConfigOption::floatOption("Threshold value.",           1.0, 0.0, 5.0));

        config.setPriorityOptions({"sigmaT", "sigmaS", "threshold"});
    };

    ReclusiveEventDenoisor() {
        sizeX = ceil((float) inputs.getEventInput("events").sizeX() / _THREAD_) * _THREAD_;
        sizeY = ceil((float) inputs.getEventInput("events").sizeY() / _THREAD_) * _THREAD_;
        _LENGTH_ = sizeX * sizeY;
	    outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

	void run() override {
        auto inEvent  = inputs.getEventInput("events").events();
        auto outEvent = outputs.getEventOutput("events").events();

        Xt = (float_t *) calloc(_POLES_ * _LENGTH_, sizeof(float_t));
        Yt = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));
        Ut = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));

        if (!inEvent) {
            return;
        }

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
                outEvent << evt;
            }
        }
        outEvent << dv::commit;

        free(Xt);
        free(Yt);
        free(Ut);
    };

	void configUpdate() override {
        sigmaT    = config.getFloat("sigmaT");
        sigmaS    = config.getFloat("sigmaS");
        threshold = config.getFloat("threshold");
        setCoefficient();
    };
};

}

registerModuleClass(kdv::ReclusiveEventDenoisor)
