// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_optimal_bs.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupOptimalBS, "MarkupOptimalBS", 0);

namespace {
size_t get_hueristic_optimal_batch(const std::shared_ptr<ov::Node>& node) {
    std::set<std::string> batch1{"resnet_model/conv2d/Conv2D",
                                 "resnet_model/conv2d_2/Conv2D",
                                 "resnet_model/conv2d_1/Conv2D"};
    std::set<std::string> batch2{"resnet_model/conv2d_6/Conv2D",
                                 "resnet_model/conv2d_14/Conv2D",
                                 "resnet_model/conv2d_11/Conv2D",
                                 "resnet_model/conv2d_15/Conv2D"};

    const auto name = node->get_friendly_name();
    if (batch1.count(name)) {
        return 1;
    }
    if (batch2.count(name)) {
        return 2;
    }
    return 0;
}
}  // namespace

ov::intel_cpu::MarkupOptimalBS::MarkupOptimalBS() {
    auto conv_m = ngraph::pattern::wrap_type<ngraph::opset1::Convolution,
                                             ngraph::opset1::ConvolutionBackpropData,
                                             ov::intel_cpu::FullyConnectedNode>(ngraph::pattern::has_static_dim(0));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();
        const size_t optimal_bs = get_hueristic_optimal_batch(node);

        const auto cur_bs = m.get_match_value().get_partial_shape()[0].get_length();
        if (cur_bs > optimal_bs) {
            ov::intel_cpu::set_optimal_bs(node, optimal_bs);
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_m, "MarkupOptimalBS");
    this->register_matcher(m, callback);
}
