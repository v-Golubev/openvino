#include <sstream>
#include <string>
#include <thread>
#include "openvino/openvino.hpp"

static ov::Core core;
static std::shared_ptr<ov::Model> model;
static std::string device_name = "CPU";

static void StartThreadForRun() {
        auto compiled_model = core.compile_model(model, device_name);
        auto xrand = []() {
                unsigned long align = 32;
                return (rand() % (2048-align) & ~(align-1)) + align;
        };
        for (;;) {
                ov::InferRequest request = compiled_model.create_infer_request();
                auto shape = ov::Shape{3, xrand(), xrand()};
                std::cout << std::this_thread::get_id() << " " << "shape: " << shape << std::endl;
                request.get_input_tensor().set_shape(shape);
                request.infer();
        }
}

int main() {
    core.set_property(device_name, ov::streams::num(0));
    core.set_property(device_name, ov::affinity(ov::Affinity::NONE));
    core.set_property(device_name, {{"CPU_RUNTIME_CACHE_CAPACITY", "0"}});
    model = core.read_model("/home/vgolubev/openvino/bin/intel64/Release/MaskRCNN-10.onnx");

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < std::thread::hardware_concurrency(); i++)
        threads.emplace_back(StartThreadForRun);
    for (auto &t : threads)
        t.join();
}