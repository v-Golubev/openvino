// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace ov {
namespace intel_cpu {
bool has_optimal_bs(const std::shared_ptr<ov::Node>& node);
size_t get_optimal_bs(const std::shared_ptr<ov::Node>& node);
void set_optimal_bs(const std::shared_ptr<ov::Node>& node, const size_t opt_batch);

class OptimalBatchSize : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("optimal_batch_size");
    OptimalBatchSize() = default;
    OptimalBatchSize(const size_t value) : value(value) {}

    const size_t get_value() { return value; }
    void set_value(const size_t _value) { value = _value; }
    std::string to_string() const override { return std::to_string(value); }
    bool operator==(const OptimalBatchSize &rhs) const { return value == rhs.value; }
    bool operator!=(const OptimalBatchSize &rhs) const { return value != rhs.value; }

private:
    size_t value = 0;
};
}  // namespace intel_cpu
}  // namespace ov
