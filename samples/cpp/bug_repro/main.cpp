#include <string>
#include <thread>
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

size_t getSystemDataByName(char* name) {
    auto parseLine = [](std::string line) -> size_t {
        std::string res = "";
        for (auto c : line)
            if (isdigit(c))
                res += c;
        if (res.empty())
            throw std::runtime_error("Can't get system memory values");
        return std::stoul(res);
    };    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    bool status = false;
    if (file != nullptr) {
        char line[128];
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, name, strlen(name)) == 0) {
                result = parseLine(line);
                status = true;
                break;
            }
        }
        fclose(file);
    }
    if (!status)
        throw std::runtime_error("Can't get system memory values");
    return result;
}

size_t getVmRSSInKB() {
    return getSystemDataByName(const_cast<char*>("VmRSS:"));
}

ov::Core core;
std::shared_ptr<ov::Model> model;
std::string device_name = "CPU";
auto xrand = []() -> size_t {
    return (rand() % 200) + 100;
};

void StartThreadForRun() {
    auto compiled_model = core.compile_model(model, device_name);
    ov::InferRequest request = compiled_model.create_infer_request();
    for (;;) {
        auto shape = ov::Shape{1, 3, xrand(), xrand()};
        std::cerr << shape << std::endl;
        auto t = request.get_input_tensor();
        t.set_shape(shape);
        request.set_input_tensor(t);
        request.infer();
        std::cout << getVmRSSInKB() << std::endl;
    }
}

int main() {
    model = core.read_model("/home/vgolubev/models/one_layer.xml");
    StartThreadForRun();
} 