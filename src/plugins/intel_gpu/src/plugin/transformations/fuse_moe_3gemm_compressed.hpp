// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Fuse routing subgraph + MOECompressed (or plain MOE with f16 weights) into MOE3GemmFusedCompressed.
// Handles both Softmax and Sigmoid+bias routing patterns via pattern::op::Or.
class FuseMOE3GemmCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOE3GemmCompressed");
    explicit FuseMOE3GemmCompressed(bool has_batch_dim = false);

private:
    bool m_has_batch_dim = false;
};

}   // namespace ov::intel_gpu
