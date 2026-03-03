// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/moe_compressed.hpp"

namespace ov::intel_gpu::op {

/// \brief MOE3GemmFusedCompressed that support compressed and fused MOE for GEMM3_SWIGLU.
class MOE3GemmFusedCompressed : public MOECompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressed", "gpu_opset", MOECompressed);

    MOE3GemmFusedCompressed() = default;

    /// \brief Constructs a MOE3GemmFusedCompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, topk] normalized weights for selected experts
    ///   2: topk_indices - [num_seq, topk] indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   4: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   6: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   7: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   9: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, group_num, group_size]
    ///   10: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///   11: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    /// \param config Configuration for the MOE 3GEMM SWIGLU fused operation
    MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
