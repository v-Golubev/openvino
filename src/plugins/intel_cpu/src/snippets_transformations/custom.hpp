// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class CustomTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CustomTransformation", "0");
    CustomTransformation();
};

class ReshapeBcastOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeBcastOptimization", "0");
    ReshapeBcastOptimization();
};
}   // namespace intel_cpu
}   // namespace ov
