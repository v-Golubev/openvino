﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/relu.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void ReluTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Relu>({ make_op_label<opset1::Multiply>()}));
}

void ReluTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> relu = m.get_match_root();
    if (!canBeTransformed(context, relu)) {
        return;
    }

    relu = separateInStandaloneBranch(relu);
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(relu, 0);
    if (dequantization.subtract == nullptr) {
        moveDequantizationAfter(context, relu, dequantization, true);
    } else {
        moveMultiplyAfter(context, relu, dequantization, true);
    }
}

bool ReluTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool ReluTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, 0);
    const std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(dequantization.multiply->input_value(1).get_node_shared_ptr());
    const auto scales = constant->cast_vector<float>();
    if (std::all_of(scales.begin(), scales.end(), [](const float value) { return value < 0.f; })) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
