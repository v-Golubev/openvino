﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/depth_to_space.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void DepthToSpaceTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::DepthToSpace>({ make_op_label<ngraph::opset1::Multiply>() }));
}

void DepthToSpaceTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    const std::shared_ptr<Node> depthToSpace = separateInStandaloneBranch(m.get_match_root());
    moveDequantizationAfter(context, depthToSpace, NetworkHelper::getDequantization(depthToSpace), true);
}

bool DepthToSpaceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    // TODO: change when getDequantization will be expanded
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer);
    if (dequantization.multiply != nullptr) {
        auto multiplyConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
        if (!NetworkHelper::isScalarLike(multiplyConst)) {
            return false;
        }
    }

    if (dequantization.subtract != nullptr) {
        auto subtractConst = as_type_ptr<opset1::Constant>(dequantization.subtract->get_input_node_shared_ptr(1));
        if (!NetworkHelper::isScalarLike(subtractConst)) {
            return false;
        }
    }

    return true;
}
