// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {

MOE3GemmFusedCompressed::MOE3GemmFusedCompressed(const OutputVector& args, const ov::op::internal::MOECompressed::Config config) : ov::op::internal::MOECompressed(args, config) {
    constructor_validate_and_infer_types();
}

void MOE3GemmFusedCompressed::validate_and_infer_types() {
    // Base input counts: SOFTMAX=13 (11 + 2 dummy bias/eps), SIGMOID_BIAS=13, shared=23.
    size_t expected_inputs = m_config.num_shared_expert > 0 ? 23 : 13;
    if (m_config.has_per_expert_scale)
        expected_inputs += 1;
    OPENVINO_ASSERT(get_input_size() == expected_inputs,
                    "MOECompressed: expected ",
                    expected_inputs,
                    " inputs for routing type ",
                    m_config.routing_type,
                    ", got ",
                    get_input_size());

    if (m_config.routing_type == MOECompressed::RoutingType::SIGMOID_BIAS) {
        // Input 12 is routing_eps — must be a scalar
        OPENVINO_ASSERT(ov::shape_size(get_input_partial_shape(12).to_shape()) == 1,
                        "MOE3GemmFusedCompressed: routing_eps (input 12) must be scalar, got shape ",
                        get_input_partial_shape(12));
    }

    // Set output type/shape. Do NOT call MOECompressed::validate_and_infer_types()
    // because the parent's weight-index validation assumes the standard MOECompressed
    // input layout (with topk_indices at index 2), which doesn't apply here.
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;
    set_output_type(0, output_type, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MOE3GemmFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOE3GemmFusedCompressed>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op
