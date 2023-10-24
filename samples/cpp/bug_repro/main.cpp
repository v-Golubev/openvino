#include <sstream>
#include <string>
#include <thread>
#include "openvino/openvino.hpp"
#include "transformations/convert_precision.hpp"

static ov::Core core;
static std::shared_ptr<ov::Model> model;
static std::string device_name = "TEMPLATE";
static const std::string plugin_name = "openvino_template_plugin";

static void StartThreadForRun() {
        auto compiled_model = core.compile_model(model, device_name);
        auto xrand = []() {
                unsigned long align = 32;
                return (rand() % (2048-align) & ~(align-1)) + align;
        };
        for (;;) {
                ov::InferRequest request = compiled_model.create_infer_request();
                auto shape = ov::Shape{3, xrand(), xrand()};
                // auto shape = ov::Shape{3, 1024, 1024};
                // std::cout << std::this_thread::get_id() << " " << "shape: " << shape << std::endl;
                request.get_input_tensor().set_shape(shape);
                request.infer();
        }
}

int main() {
        core.register_plugin(plugin_name, device_name);
        core.set_property(device_name, ov::streams::num(0));
        core.set_property(device_name, ov::affinity(ov::Affinity::NONE));
        model = core.read_model("/home/lc/golubevv/openvino/samples/cpp/bug_repro/MaskRCNN-12.onnx");
        
        precisions_map precisions = {
            {ov::element::i64,     ov::element::i32},
            {ov::element::u64,     ov::element::i32},
            {ov::element::i16,     ov::element::i32},
            {ov::element::u16,     ov::element::i32},
            {ov::element::u32,     ov::element::i32},
        };
        ov::pass::ConvertPrecision(precisions).run_on_model(model);
        model->validate_nodes_and_infer_types();

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < std::thread::hardware_concurrency(); i++)
                threads.emplace_back(StartThreadForRun);
        for (auto &t : threads)
                t.join();
}