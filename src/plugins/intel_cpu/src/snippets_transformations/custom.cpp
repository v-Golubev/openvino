// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>

#include "itt.hpp"

using namespace ngraph;

namespace ov {
namespace intel_cpu {
// CustomTransformation::CustomTransformation() {
//     MATCHER_SCOPE(CustomTransformation);
//     auto full_input_m = pattern::any_input(pattern::rank_equals(3));
//     auto reshape_m = pattern::wrap_type<opset1::Reshape>({full_input_m, pattern::wrap_type<opset1::Constant>()});
//     auto sparse_input_m = pattern::any_input(pattern::has_static_shape());
//     auto add_m = pattern::wrap_type<opset1::Add>({reshape_m, sparse_input_m});

//     ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
//         const auto& pattern_map = m.get_pattern_value_map();
//         const auto& full_input = pattern_map.at(full_input_m);
//         const auto& sparse_input = pattern_map.at(sparse_input_m);
//         const auto& reshape = pattern_map.at(reshape_m).get_node_shared_ptr();

//         const auto& full_shape = reshape->get_output_shape(0);
//         const auto& sparse_shape = sparse_input.get_shape();
//         if (ov::shape_size(sparse_shape) >= ov::shape_size(full_shape) || sparse_shape.size() != full_shape.size()) {
//             return false;
//         }

//         const auto add = pattern_map.at(add_m).get_node_shared_ptr();
//         const auto bcast_shape = ov::opset1::Constant::create(ov::element::i32, {full_shape.size()}, full_shape);
//         const auto bcast = std::make_shared<ov::opset1::Broadcast>(sparse_input, bcast_shape);

//         const auto& full_shape_before = full_input.get_shape();
//         const auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {full_shape_before.size()}, full_shape_before);
//         const auto sparse_reshape = std::make_shared<ov::opset1::Reshape>(bcast, reshape_const, true);
//         ov::copy_runtime_info(sparse_input.get_node_shared_ptr(), {bcast, sparse_reshape});

//         const auto new_add = add->clone_with_new_inputs({full_input, sparse_reshape});
//         const auto new_reshape = reshape->clone_with_new_inputs({new_add, reshape->input_value(1)});
//         new_reshape->set_friendly_name(add->get_friendly_name());
//         ov::copy_runtime_info({add, reshape}, {new_add, new_reshape});

//         if (ov::replace_node_update_name(add, new_reshape)) {
//             std::cout << "Reshape prop transformation...\n\t" << add << std::endl;
//         }
//         return true;
//     };

//     auto m = std::make_shared<ngraph::pattern::Matcher>(add_m, matcher_name);
//     register_matcher(m, callback);
// }

CustomTransformation::CustomTransformation() {
    MATCHER_SCOPE(CustomTransformation);
    auto input_m = pattern::any_input(pattern::rank_equals(3));
    auto reshape_1_m = pattern::wrap_type<opset1::Reshape>({input_m, pattern::wrap_type<opset1::Constant>()});
    auto sparse_input_1_m = pattern::any_input(pattern::rank_equals(5));
    auto sparse_input_2_m = pattern::any_input(pattern::rank_equals(5));
    auto add_1_m = pattern::wrap_type<opset1::Add>({reshape_1_m, sparse_input_1_m});
    auto add_2_m = pattern::wrap_type<opset1::Add>({add_1_m, sparse_input_2_m});
    auto reshape_2_m = pattern::wrap_type<opset1::Reshape>({add_2_m, pattern::wrap_type<opset1::Constant>()}, pattern::rank_equals(3));

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& input = pattern_map.at(input_m);
        const auto& sparse_input_1 = pattern_map.at(sparse_input_1_m);
        const auto& sparse_input_2 = pattern_map.at(sparse_input_2_m);

        const auto new_add = std::make_shared<ov::opset1::Add>(sparse_input_1, sparse_input_2);
        const auto& in_shape = input.get_shape();
        const auto target_shape = ov::opset1::Constant::create(ov::element::i32, {in_shape.size()}, in_shape);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(new_add, target_shape, true);
        const auto main_add = std::make_shared<ov::opset1::Add>(input, reshape);

        const auto& old_reshape = pattern_map.at(reshape_2_m);
        if (ov::replace_output_update_name(old_reshape, main_add)) {
            std::cout << "Reshape hack transformation...\n\t" << old_reshape.get_node()->get_friendly_name() << std::endl;
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_2_m, matcher_name);
    register_matcher(m, callback);
}

ReshapeBcastOptimization::ReshapeBcastOptimization() {
    MATCHER_SCOPE(ReshapeBcastOptimization);
    auto input_m = pattern::any_input(pattern::rank_equals(4));
    auto reshape_const_m = pattern::wrap_type<opset1::Constant>();
    auto reshape_m = pattern::wrap_type<opset1::Reshape>({input_m, reshape_const_m}, pattern::rank_equals(5));
    auto bcast_const_m = pattern::wrap_type<opset1::Constant>();
    auto bcast_m = pattern::wrap_type<opset1::Broadcast>({reshape_m, bcast_const_m, pattern::any_input()}, pattern::rank_equals(5));

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape = pattern_map.at(reshape_m).get_node_shared_ptr();
        const auto bcast = pattern_map.at(bcast_m).get_node_shared_ptr();
        const auto reshape_const = ov::as_type_ptr<ov::opset1::Constant>(pattern_map.at(reshape_const_m).get_node_shared_ptr());
        const auto bcast_const = ov::as_type_ptr<ov::opset1::Constant>(pattern_map.at(bcast_const_m).get_node_shared_ptr());

        const auto reshape_vals = reshape_const->cast_vector<int>();
        const auto bcast_vals = bcast_const->cast_vector<int>();

        if (reshape_vals.size() != bcast_vals.size()) {
            return false;
        }

        const size_t size = reshape_vals.size();
        for (size_t i = 0; i < size - 1; ++i) {
            if (reshape_vals[i] != bcast_vals[i]) {
                return false;
            }
        }
        if (reshape_vals.back() != 1) {
            return false;
        }

        auto zero_const = ov::opset1::Constant::create(ov::element::f32, bcast->get_output_shape(0), {0.f});
        const auto add = std::make_shared<ov::opset1::Add>(reshape->output(0), zero_const);

        if (ov::replace_node_update_name(bcast, add)) {
            std::cout << "Bcast-Reshape transformation...\n\t" << reshape << std::endl;
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bcast_m, matcher_name);
    register_matcher(m, callback);
}
}   // namespace intel_cpu
}   // namespace ov
