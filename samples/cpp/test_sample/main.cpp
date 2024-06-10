#include <sstream>
#include <string>
#include <thread>
#include "openvino/openvino.hpp"

static ov::Core core;
static std::shared_ptr<ov::Model> model;
static std::string device_name = "CPU";

static void StartThreadForRun() {
    auto xrand = []() -> size_t {
        unsigned long align = 32;
        return (rand() % (2048-align) & ~(align-1)) + align;
    };
    auto compiled_model = core.compile_model(model, device_name);
    for (size_t i = 0; true; ++i) {
        ov::InferRequest request = compiled_model.create_infer_request();
        auto shape = ov::Shape{1, 3, xrand(), xrand()};
        request.get_input_tensor().set_shape(shape);
        request.infer();
    }
}

int main(int argc, char** argv) {
    const size_t threads_num = std::thread::hardware_concurrency();
    std::cout << "threads num = " << threads_num << std::endl;
    core.set_property(device_name, ov::streams::num(0));
    core.set_property(device_name, {{"CPU_RUNTIME_CACHE_CAPACITY", "0"}});

    const std::string model_path = argv[1];
    model = core.read_model(model_path);
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < threads_num; i++)
        threads.emplace_back(StartThreadForRun);
    for (auto &t : threads)
        t.join();
}