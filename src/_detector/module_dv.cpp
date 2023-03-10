#include "selective_event_detector.hpp"

namespace kdv {

    class SelectiveDetector : public edt::SelectiveDetector, public dv::ModuleBase {
    private:
        static int getColorHex(const std::string &hexColor) {
            auto c = std::stoi(hexColor, nullptr, 16);
            if ((c < 0) || (c > 255)) {
                c = 0;
            }
            return c;
        }

        dv::EventStreamSlicer slicer;
        cv::Mat frame;
        cv::Vec3b backgroundColor;
        cv::Vec3b positiveColor;
        cv::Vec3b negativeColor;
        int8_t lineWidth = 2;
        int64_t lastTimestamp{0};

        void renderFrame(const dv::EventStore &inEvent) {
            frame = backgroundColor;

            binaryImg = cv::Mat::zeros(sizeY, sizeX, CV_8UC1);
            for (const auto &evt : inEvent) {
                frame.at<cv::Vec3b>(evt.y(), evt.x())   = evt.polarity() ? positiveColor : negativeColor;
                binaryImg.at<uint8_t>(evt.y(), evt.x()) = 255;
            }

            findContoursRect(binaryImg);
            auto rankedRect = selectiveBoundingBox();

            cv::RNG_MT19937 rng(1225);
            for (size_t i = 0; i < rankedRect.size(); i++) {
                auto rect = rankedRect[i].second;
                if (i > maxRectNum)
                    break;
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::rectangle(frame, rect.tl(), rect.br(), color, lineWidth);
            }

            auto outFrame = outputs.getFrameOutput("frames").frame();
            outFrame.setTimestamp((inEvent.isEmpty()) ? (lastTimestamp) : (inEvent.getLowestTime()));
            outFrame.setExposure(std::chrono::milliseconds{static_cast<int64_t>(1000.0f / 30.0f)});
            outFrame.setPosition(0, 0);
            outFrame.setSource(dv::FrameSource::VISUALIZATION);
            outFrame.setMat(frame);
            outFrame.commit();
        }

    public:
        static const char *initDescription() {
            return "Selective fire detector.";
        };

        static void initInputs(dv::InputDefinitionList &in) {
            in.addEventInput("events");
        }

        static void initOutputs(dv::OutputDefinitionList &out) {
            out.addFrameOutput("frames");
        }

        static void initConfigOptions(dv::RuntimeConfig &config) {
            config.add("backgroundColor", dv::ConfigOption::stringOption("Background color in hex format #RRGGBB.", "000000", 6, 6));
            config.add("positiveColor", dv::ConfigOption::stringOption("Positive (ON) event color in hex format #RRGGBB.", "FF0000", 6, 6));
            config.add("negativeColor", dv::ConfigOption::stringOption("Negative (OFF) event color in hex format #RRGGBB.", "00FF00", 6, 6));
            config.add("lineWidth", dv::ConfigOption::intOption("Max rectangle number.", 2, 1, 5));

            config.add("maxRectNum", dv::ConfigOption::intOption("Max rectangle number.", 5, 1, 10));
            config.add("minRectArea", dv::ConfigOption::intOption("Min rectangle area.", 10, 1, 100));
            config.add("threshold", dv::ConfigOption::floatOption("Threshold.", 0.85, 0.0, 1.0));
            config.setPriorityOptions({"maxRectNum", "minRectArea", "threshold"});
        };

        SelectiveDetector() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;

            slicer.doEveryTimeInterval(std::chrono::milliseconds{static_cast<int64_t>(1000.0f / 30.0f)},
                                       std::function<void(const dv::EventStore &)>(std::bind(&SelectiveDetector::renderFrame, this, std::placeholders::_1)));

            frame = cv::Mat{inputs.getEventInput("events").size(), CV_8UC3};

            outputs.getFrameOutput("frames").setup(inputs.getEventInput("events"));
        };

        void run() override {
            slicer.accept(inputs.getEventInput("events").events());
        };

        void configUpdate() override {
            maxRectNum  = config.getInt("maxRectNum");
            minRectArea = config.getInt("minRectArea");
            threshold   = config.getFloat("threshold");

            lineWidth          = config.getInt("lineWidth");
            backgroundColor(0) = getColorHex(config.getString("backgroundColor").substr(4, 2));
            backgroundColor(1) = getColorHex(config.getString("backgroundColor").substr(2, 2));
            backgroundColor(2) = getColorHex(config.getString("backgroundColor").substr(0, 2));

            positiveColor(0) = getColorHex(config.getString("positiveColor").substr(4, 2));
            positiveColor(1) = getColorHex(config.getString("positiveColor").substr(2, 2));
            positiveColor(2) = getColorHex(config.getString("positiveColor").substr(0, 2));

            negativeColor(0) = getColorHex(config.getString("negativeColor").substr(4, 2));
            negativeColor(1) = getColorHex(config.getString("negativeColor").substr(2, 2));
            negativeColor(2) = getColorHex(config.getString("negativeColor").substr(0, 2));
        }
    };

}

registerModuleClass(kdv::SelectiveDetector)
