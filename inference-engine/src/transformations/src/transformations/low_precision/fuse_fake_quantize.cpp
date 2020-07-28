﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fuse_fake_quantize.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseFakeQuantizeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FuseFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(m.get_match_root());
    const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(fakeQuantize->shared_from_this());
    if (dequantization.empty()) {
        return;
    }

    std::shared_ptr<Node> parent;
    std::shared_ptr<Node> inputLowConst = fakeQuantize->get_input_node_shared_ptr(1);
    std::shared_ptr<Node> inputHightConst = fakeQuantize->get_input_node_shared_ptr(2);

    if (dequantization.multiply != nullptr) {
        inputLowConst = fold<opset1::Divide>(inputLowConst, dequantization.multiply->get_input_node_shared_ptr(1));
        inputHightConst = fold<opset1::Divide>(inputHightConst, dequantization.multiply->get_input_node_shared_ptr(1));

        Shape shape = inputLowConst->get_output_shape(0);
        if ((shape.size() < fakeQuantize->get_output_shape(0).size()) && (shape.size() > 1ul)) {
            inputLowConst = fold<opset1::Unsqueeze>(
                inputLowConst,
                std::make_shared<opset1::Constant>(
                    ngraph::element::i32,
                    Shape{ 1 },
                    std::vector<size_t>({ 0ul })));

            inputHightConst = fold<opset1::Unsqueeze>(
                inputHightConst,
                std::make_shared<opset1::Constant>(
                    ngraph::element::i32,
                    Shape{ 1 },
                    std::vector<size_t>({ 0ul })));
        }
        parent = dequantization.multiply->get_input_node_shared_ptr(0);
    }

    if (dequantization.subtract != nullptr) {
        inputLowConst = fold<opset1::Add>(inputLowConst, dequantization.subtract->get_input_node_shared_ptr(1));
        inputHightConst = fold<opset1::Add>(inputHightConst, dequantization.subtract->get_input_node_shared_ptr(1));
        parent = dequantization.subtract->get_input_node_shared_ptr(0);
    }

    if (dequantization.convert != nullptr) {
        parent = dequantization.convert->get_input_node_shared_ptr(0);
    }

    std::shared_ptr<Node> newFakeQuantize = fakeQuantize->clone_with_new_inputs({
        parent,
        inputLowConst,
        inputHightConst,
        fakeQuantize->input_value(3),
        fakeQuantize->input_value(4) });

    replace_node(fakeQuantize, newFakeQuantize);

    updateOutput(context, newFakeQuantize, fakeQuantize);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
