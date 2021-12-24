// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <openvino/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::SkipGatherBeforeTransposeAndReshape, "SkipGatherBeforeTransposeAndReshape", 0);

ngraph::pass::SkipGatherBeforeTransposeAndReshape::SkipGatherBeforeTransposeAndReshape() {
    MATCHER_SCOPE(SkipGatherBeforeTransposeAndReshape);

    auto input_m = ngraph::pattern::any_input();
    auto gather_m = ngraph::pattern::wrap_type<ngraph::op::util::GatherBase>({input_m,
                                                            ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
                                                            ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto transpose_m = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({gather_m, ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto reshape_m = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({transpose_m, ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& input = pattern_map.at(input_m);
        if (input.get_partial_shape()[0] != 1) {
            return false;
        }

        const auto& gather = pattern_map.at(gather_m).get_node_shared_ptr();
        const auto indices_node = as_type_ptr<ngraph::opset8::Constant>(gather->get_input_node_shared_ptr(1));
        const auto axis_node = as_type_ptr<ngraph::opset8::Constant>(gather->get_input_node_shared_ptr(2));
        if (!indices_node || !axis_node) {
            return false;
        }

        const std::vector<std::int64_t> expected_gather_value{0};
        if (indices_node->cast_vector<std::int64_t>() != expected_gather_value ||
            axis_node->cast_vector<std::int64_t>() != expected_gather_value) {
            return false;
        }

        const auto& transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
        const auto transpose_const = as_type_ptr<ngraph::opset8::Constant>(transpose->get_input_node_shared_ptr(1));
        if (!transpose_const) {
            return false;
        }

        const auto& reshape = pattern_map.at(reshape_m).get_node_shared_ptr();
        const auto reshape_const = as_type_ptr<ngraph::opset8::Constant>(reshape->get_input_node_shared_ptr(1));
        if (!reshape_const) {
            return false;
        }

        const auto transpose_vals = transpose_const->cast_vector<std::int64_t>();
        std::vector<std::int64_t> new_transpose_vals{0};
        for (auto elem : transpose_vals) {
            new_transpose_vals.push_back(++elem);
        }

        const auto new_transpose_const = ngraph::opset8::Constant::create(transpose_const->get_element_type(),
                                                                          {new_transpose_vals.size()},
                                                                          new_transpose_vals);
        const auto new_transpose = transpose->clone_with_new_inputs({input, new_transpose_const});
        new_transpose->set_friendly_name(transpose->get_friendly_name());
        ngraph::copy_runtime_info(transpose, new_transpose);
        ngraph::replace_node(transpose, new_transpose);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_m, matcher_name);
    register_matcher(m, callback);
}
