// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

#include <cstring>
#include <initializer_list>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using namespace ov::pass;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

// Note: intermediate nodes remain unchanged,
// but we need to explicitly call shape inference for them to keep shapes consistency
void validate_nodes(const pattern::PatternValueMap& map, const std::initializer_list<std::shared_ptr<ov::Node>> nodes) {
    for (const auto& node : nodes) {
        map.at(node).get_node_shared_ptr()->validate_and_infer_types();
    }
};

// Concatenate two Constants along the given axis via raw byte-level copy.
// Handles sub-byte types (u4/i4) correctly by requiring byte-aligned slices.
// Falls back to make_try_fold<Concat> for standard (byte-aligned) element types
// or when byte alignment cannot be guaranteed.
std::shared_ptr<ov::Node> concat_constants_raw(const std::shared_ptr<v0::Constant>& c1,
                                               const std::shared_ptr<v0::Constant>& c2,
                                               int64_t axis) {
    const auto& et = c1->get_element_type();
    const size_t bitwidth = et.bitwidth();

    // For standard byte-sized types, use make_try_fold (well-tested path)
    if (bitwidth >= 8) {
        return ov::op::util::make_try_fold<v0::Concat>(ov::OutputVector{c1, c2}, axis);
    }

    // Sub-byte types: manual byte-level copy to avoid potential evaluation bugs
    auto s1 = c1->get_shape();
    auto s2 = c2->get_shape();
    const size_t rank = s1.size();
    const size_t pos_axis = axis >= 0 ? static_cast<size_t>(axis) : rank + axis;
    if (pos_axis >= rank || s1.size() != s2.size()) {
        return nullptr;
    }

    // Validate shapes match on all axes except the concat axis
    for (size_t i = 0; i < rank; ++i) {
        if (i != pos_axis && s1[i] != s2[i]) {
            return nullptr;
        }
    }

    auto out_shape = s1;
    out_shape[pos_axis] += s2[pos_axis];

    size_t outer = 1;
    for (size_t i = 0; i < pos_axis; ++i)
        outer *= s1[i];

    size_t inner = 1;
    for (size_t i = pos_axis + 1; i < rank; ++i)
        inner *= s1[i];

    const size_t s1_slice_elems = s1[pos_axis] * inner;
    const size_t s2_slice_elems = s2[pos_axis] * inner;
    const size_t elems_per_byte = 8 / bitwidth;

    // Require byte-aligned slices for correct sub-byte copy
    if (s1_slice_elems % elems_per_byte != 0 || s2_slice_elems % elems_per_byte != 0) {
        // Fall back to make_try_fold (may be buggy for sub-byte but no better option)
        return ov::op::util::make_try_fold<v0::Concat>(ov::OutputVector{c1, c2}, axis);
    }

    const size_t s1_bytes = s1_slice_elems * bitwidth / 8;
    const size_t s2_bytes = s2_slice_elems * bitwidth / 8;

    ov::Tensor out_tensor(et, out_shape);
    auto* src1 = static_cast<const uint8_t*>(c1->get_data_ptr());
    auto* src2 = static_cast<const uint8_t*>(c2->get_data_ptr());
    auto* dst = static_cast<uint8_t*>(out_tensor.data());

    for (size_t o = 0; o < outer; ++o) {
        std::memcpy(dst, src1, s1_bytes);
        dst += s1_bytes;
        src1 += s1_bytes;
        std::memcpy(dst, src2, s2_bytes);
        dst += s2_bytes;
        src2 += s2_bytes;
    }

    return std::make_shared<v0::Constant>(out_tensor);
}

std::shared_ptr<ov::op::v0::Unsqueeze> introduce_n_experts_dim(const ov::Output<ov::Node>& data) {
    auto zero_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, zero_const);
    ov::copy_runtime_info(data.get_node_shared_ptr(), {unsqueeze, zero_const});
    return unsqueeze;
}

// --- Shared pattern node containers ---

struct MOE2GEMMPatternNodes {
    std::shared_ptr<ov::Node> experts_input, tile, after_tile_reshape;
    std::shared_ptr<ov::Node> gate_up_matmul, gate_up_add, gate_up_bias;
    std::shared_ptr<ov::Node> slice1, clamp, add1, slice2, minimum1, swish_beta, swish, multiply2;
    std::shared_ptr<ov::Node> down_proj_matmul, down_proj_bias, down_proj_add;
    std::shared_ptr<ov::Node> end_reshape_target_shape, end_reshape;
    std::shared_ptr<ov::Node> router_topk_indices, chosen_experts, scatter_elements_update;
    std::shared_ptr<ov::Node> router_transpose, router_reshape, optional_unsqueeze;
    std::shared_ptr<ov::Node> mul3, reduce_sum;
};

MOE2GEMMPatternNodes build_2gemm_pattern() {
    MOE2GEMMPatternNodes p;

    p.experts_input = pattern::wrap_type<v1::Reshape>({pattern::any_input(), pattern::any_input()});
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()});
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()});
    p.gate_up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.gate_up_bias = pattern::wrap_const();
    p.gate_up_add = pattern::wrap_type<v1::Add>({p.gate_up_matmul, p.gate_up_bias}, pattern::consumers_count(2));

    // Branch 1: Slice_1 -> Clamp -> Add_1
    p.slice1 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    p.clamp = pattern::wrap_type<v0::Clamp>({p.slice1});
    p.add1 = pattern::wrap_type<v1::Add>({p.clamp, pattern::wrap_const()});

    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    p.slice2 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    p.minimum1 = pattern::wrap_type<v1::Minimum>({p.slice2, pattern::wrap_const()});
    p.swish_beta = pattern::wrap_const();
    p.swish = pattern::wrap_type<v4::Swish>({p.minimum1, p.swish_beta});

    // Join: Multiply_2
    p.multiply2 = pattern::wrap_type<v1::Multiply>({p.add1, p.swish});

    // Down projection
    p.down_proj_matmul = pattern::wrap_type<v0::MatMul>(
        {p.multiply2, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.down_proj_bias = pattern::wrap_const();
    p.down_proj_add = pattern::wrap_type<v1::Add>({p.down_proj_matmul, p.down_proj_bias});
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape = pattern::wrap_type<v1::Reshape>({p.down_proj_add, p.end_reshape_target_shape});

    // Routing weights/mask
    p.router_topk_indices = pattern::any_input();
    p.chosen_experts = pattern::any_input();
    p.scatter_elements_update = pattern::wrap_type<v3::ScatterElementsUpdate, v12::ScatterElementsUpdate>(
        {pattern::any_input(), p.router_topk_indices, p.chosen_experts, pattern::any_input()});

    p.router_transpose = pattern::wrap_type<v1::Transpose>({p.scatter_elements_update, pattern::any_input()});
    p.router_reshape = pattern::wrap_type<v1::Reshape>({p.router_transpose, pattern::any_input()});
    p.optional_unsqueeze = pattern::optional<v0::Unsqueeze>({p.router_reshape, pattern::any_input()});

    p.mul3 = pattern::wrap_type<v1::Multiply>({p.end_reshape, p.optional_unsqueeze});
    p.reduce_sum = pattern::wrap_type<v1::ReduceSum>({p.mul3, pattern::any_input()}, {{"keep_dims", false}});

    return p;
}

struct MOE3GEMMPatternNodes {
    std::shared_ptr<ov::Node> experts_input, tile, after_tile_reshape;
    std::shared_ptr<ov::Node> gate_matmul, swish, up_matmul, swiglu;
    std::shared_ptr<ov::Node> down_matmul;
    std::shared_ptr<ov::Node> end_reshape_target_shape, end_reshape;
    std::shared_ptr<ov::Node> router_topk_indices, chosen_experts, scatter_elements_update;
    std::shared_ptr<ov::Node> router_transpose, router_reshape, optional_unsqueeze;
    std::shared_ptr<ov::Node> mul3, reduce_sum;
};

MOE3GEMMPatternNodes build_3gemm_pattern() {
    MOE3GEMMPatternNodes p;

    p.experts_input = pattern::wrap_type<v1::Reshape>({pattern::any_input(), pattern::any_input()});
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()});
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()});

    // First GEMM (activation gate)
    p.gate_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.swish = pattern::wrap_type<v4::Swish>({p.gate_matmul});
    // Second GEMM (up_projection)
    p.up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    // Join: Multiply (SwiGLU)
    p.swiglu = pattern::wrap_type<v1::Multiply>({p.swish, p.up_matmul});

    // Third GEMM (down_projection)
    p.down_matmul = pattern::wrap_type<v0::MatMul>(
        {p.swiglu, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape = pattern::wrap_type<v1::Reshape>({p.down_matmul, p.end_reshape_target_shape});

    // Routing weights/mask
    p.router_topk_indices = pattern::any_input();
    p.chosen_experts = pattern::any_input();
    p.scatter_elements_update = pattern::wrap_type<v3::ScatterElementsUpdate, v12::ScatterElementsUpdate>(
        {pattern::any_input(), p.router_topk_indices, p.chosen_experts, pattern::any_input()});
    p.router_transpose = pattern::wrap_type<v1::Transpose>({p.scatter_elements_update, pattern::any_input()});
    p.router_reshape = pattern::wrap_type<v1::Reshape>({p.router_transpose, pattern::any_input()});
    p.optional_unsqueeze = pattern::optional<v0::Unsqueeze>({p.router_reshape, pattern::any_input()});

    p.mul3 = pattern::wrap_type<v1::Multiply>({p.end_reshape, p.optional_unsqueeze});
    p.reduce_sum = pattern::wrap_type<v1::ReduceSum>({p.mul3, pattern::any_input()}, {{"keep_dims", false}});

    return p;
}

}  // namespace

namespace ov::pass {

using ov::op::internal::GatherMatmul;

// ============================================================================
// BGM-producing passes (IR → GatherMatmul)
// ============================================================================

ConvertTiledMoeBlockTo2GatherMatmuls::ConvertTiledMoeBlockTo2GatherMatmuls() {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo2GatherMatmuls);

    auto p = build_2gemm_pattern();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& experts_subgraph_input = pm.at(p.experts_input);
        const auto& active_indices = pm.at(p.router_topk_indices);

        const auto gate_up_mm_node = pm.at(p.gate_up_matmul).get_node_shared_ptr();
        const auto gate_up_add_node = pm.at(p.gate_up_add).get_node_shared_ptr();
        const auto gate_up_bias_node = pm.at(p.gate_up_bias).get_node_shared_ptr();

        // GatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        // Number of activated experts is always 1 for the first GatherMatmul
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);
        const auto gate_up_gathered_mm = std::make_shared<GatherMatmul>(unsqueeze,
                                                                        gate_up_mm_node->input_value(1),
                                                                        active_indices,
                                                                        gate_up_bias_node);
        ov::replace_node_update_name(gate_up_add_node, gate_up_gathered_mm);

        validate_nodes(pm, {p.slice1, p.clamp, p.add1, p.slice2, p.minimum1, p.swish, p.multiply2});

        const auto down_proj_mm_node = pm.at(p.down_proj_matmul).get_node_shared_ptr();
        const auto down_proj_bias_node = pm.at(p.down_proj_bias).get_node_shared_ptr();

        const auto down_gathered_mm = std::make_shared<GatherMatmul>(pm.at(p.multiply2),
                                                                     down_proj_mm_node->input_value(1),
                                                                     active_indices,
                                                                     down_proj_bias_node);
        ov::copy_runtime_info(down_proj_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_proj_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pm.at(p.chosen_experts);
        const auto router_transpose_node = pm.at(p.router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            std::make_shared<v1::Transpose>(chosen_experts_input, router_transpose_node->input_value(1));
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        // Disconnect ScatterElementsUpdate from chosen_experts (Divide) so downstream
        // passes see correct consumers_count on both Divide and new_router_transpose.
        // Use a dummy constant to avoid giving new_router_transpose an extra dead consumer.
        auto scat_dummy = v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        pm.at(p.scatter_elements_update).get_node_shared_ptr()->input(2).replace_source_output(scat_dummy->output(0));

        const auto router_unsqueeze_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze = std::make_shared<v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pm.at(p.mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());

        const auto reduce_sum_node = pm.at(p.reduce_sum).get_node_shared_ptr();
        const auto new_reduce_sum =
            reduce_sum_node->clone_with_new_inputs({new_final_mul->output(0), reduce_sum_node->input_value(1)});
        ov::copy_runtime_info(reduce_sum_node, new_reduce_sum);
        new_reduce_sum->set_friendly_name(reduce_sum_node->get_friendly_name());

        const auto& end_reshape_out = pm.at(p.end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pm.at(p.end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto shape_slice = std::make_shared<v8::Slice>(
            end_reshape_shape,
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {shape_slice,
                               shape_slice->get_input_node_shared_ptr(1),
                               shape_slice->get_input_node_shared_ptr(2),
                               shape_slice->get_input_node_shared_ptr(3),
                               shape_slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<v1::Reshape>(new_reduce_sum, shape_slice, true);
        ov::replace_output_update_name(pm.at(p.reduce_sum), reshape->output(0));
        // To avoid friendly name duplication
        reshape->set_friendly_name(reshape->get_friendly_name() + "_Reshape");
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(p.reduce_sum, matcher_name);
    this->register_matcher(matcher, callback);
}

ConvertTiledMoeBlockTo3GatherMatmuls::ConvertTiledMoeBlockTo3GatherMatmuls() {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo3GatherMatmuls);

    auto p = build_3gemm_pattern();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& experts_subgraph_input = pm.at(p.experts_input);
        const auto& active_indices = pm.at(p.router_topk_indices);

        const auto gate_mm_node = pm.at(p.gate_matmul).get_node_shared_ptr();
        const auto up_mm_node = pm.at(p.up_matmul).get_node_shared_ptr();
        const auto down_mm_node = pm.at(p.down_matmul).get_node_shared_ptr();

        // Concatenate gate_w and up_w along axis 1
        // gate_w: [n_experts, intermediate_size, hidden_size]
        // up_w:   [n_experts, intermediate_size, hidden_size]
        // result: [n_experts, 2*intermediate_size, hidden_size]
        auto gate_w = gate_mm_node->input_value(1);
        auto up_w = up_mm_node->input_value(1);

        // Get intermediate_size from gate_w partial shape before concat
        const auto gate_w_pshape = gate_w.get_partial_shape();
        if (gate_w_pshape.rank().is_dynamic() || gate_w_pshape.rank().get_length() < 2 ||
            gate_w_pshape[1].is_dynamic()) {
            return false;
        }
        const int64_t intermediate_size = static_cast<int64_t>(gate_w_pshape[1].get_length());

        // Try constant folding; if inputs are not constants, keep a plain Concat.
        // GatherMatmul supports B on any constant-rooted path (not just v0::Constant).
        auto gate_up_w = ov::op::util::make_try_fold<v0::Concat>(ov::OutputVector{gate_w, up_w}, 1);
        ov::copy_runtime_info({gate_mm_node, up_mm_node}, gate_up_w);

        // GatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);

        // gate_up_bgm: single BGM with concatenated gate+up weights (no bias → 3-input ctor)
        const auto gate_up_gathered_mm =
            std::make_shared<GatherMatmul>(unsqueeze, gate_up_w, active_indices);
        ov::copy_runtime_info({gate_mm_node, up_mm_node}, gate_up_gathered_mm);
        gate_up_gathered_mm->set_friendly_name(gate_mm_node->get_friendly_name() + "_gate_up");

        // Slice the gate_up_bgm output to recover gate and up halves along the last axis

        // gate_slice: [..., 0:intermediate_size]
        auto gate_slice_start = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto gate_slice_stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {intermediate_size});
        auto gate_slice_step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto gate_slice_axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto gate_slice = std::make_shared<v8::Slice>(gate_up_gathered_mm,
                                                      gate_slice_start,
                                                      gate_slice_stop,
                                                      gate_slice_step,
                                                      gate_slice_axes);
        ov::copy_runtime_info(gate_mm_node,
                              {gate_slice, gate_slice_start, gate_slice_stop, gate_slice_step, gate_slice_axes});

        // up_slice: [..., intermediate_size:2*intermediate_size]
        auto up_slice_start = v0::Constant::create(ov::element::i64, ov::Shape{1}, {intermediate_size});
        auto up_slice_stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2 * intermediate_size});
        auto up_slice_step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto up_slice_axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto up_slice = std::make_shared<v8::Slice>(gate_up_gathered_mm,
                                                    up_slice_start,
                                                    up_slice_stop,
                                                    up_slice_step,
                                                    up_slice_axes);
        ov::copy_runtime_info(up_mm_node,
                              {up_slice, up_slice_start, up_slice_stop, up_slice_step, up_slice_axes});

        // Reconnect: Swish(gate_slice) * up_slice
        const auto swish_node = pm.at(p.swish).get_node_shared_ptr();
        const auto new_swish = swish_node->clone_with_new_inputs({gate_slice->output(0)});
        ov::copy_runtime_info(swish_node, new_swish);
        new_swish->set_friendly_name(swish_node->get_friendly_name());

        const auto swiglu_node = pm.at(p.swiglu).get_node_shared_ptr();
        const auto new_swiglu = swiglu_node->clone_with_new_inputs({new_swish->output(0), up_slice->output(0)});
        ov::copy_runtime_info(swiglu_node, new_swiglu);
        new_swiglu->set_friendly_name(swiglu_node->get_friendly_name());

        // down_bgm
        const auto down_gathered_mm =
            std::make_shared<GatherMatmul>(new_swiglu, down_mm_node->input_value(1), active_indices);
        ov::copy_runtime_info(down_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pm.at(p.chosen_experts);
        const auto router_transpose_node = pm.at(p.router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            std::make_shared<v1::Transpose>(chosen_experts_input, router_transpose_node->input_value(1));
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        // Disconnect ScatterElementsUpdate from chosen_experts (Divide) so downstream
        // passes see correct consumers_count on both Divide and new_router_transpose.
        // Use a dummy constant to avoid giving new_router_transpose an extra dead consumer.
        auto scat_dummy = v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        pm.at(p.scatter_elements_update).get_node_shared_ptr()->input(2).replace_source_output(scat_dummy->output(0));

        const auto router_unsqueeze_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze = std::make_shared<v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pm.at(p.mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());
        const auto reduce_sum_node = pm.at(p.reduce_sum).get_node_shared_ptr();
        const auto new_reduce_sum =
            reduce_sum_node->clone_with_new_inputs({new_final_mul->output(0), reduce_sum_node->input_value(1)});
        ov::copy_runtime_info(reduce_sum_node, new_reduce_sum);
        new_reduce_sum->set_friendly_name(reduce_sum_node->get_friendly_name());

        const auto& end_reshape_out = pm.at(p.end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pm.at(p.end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto shape_slice = std::make_shared<v8::Slice>(
            end_reshape_shape,
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {shape_slice,
                               shape_slice->get_input_node_shared_ptr(1),
                               shape_slice->get_input_node_shared_ptr(2),
                               shape_slice->get_input_node_shared_ptr(3),
                               shape_slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<v1::Reshape>(new_reduce_sum, shape_slice, true);
        ov::replace_output_update_name(pm.at(p.reduce_sum), reshape->output(0));
        reshape->set_friendly_name(new_reduce_sum->get_friendly_name() + "_Reshape");

        // Register GatherMatmul nodes so FuseConcatIntoGatherMatmulWeights
        // (in the same GraphRewrite) can visit them.
        register_new_node(gate_up_gathered_mm);
        register_new_node(down_gathered_mm);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(p.reduce_sum, matcher_name);
    this->register_matcher(matcher, callback);
}

// ============================================================================
// FuseConcatIntoGatherMatmulWeights
//
// Matches GatherMatmul where input B is a Concat of two parallel
// decompression (or constant) subgraphs. Pushes the Concat down through
// element-wise ops to the leaf Constants, eliminating the top-level Concat
// and producing a single merged chain.
//
// Before:  Const_1 → [Convert → Sub → Mul → Reshape → Convert] ──┐
//          Const_2 → [Convert → Sub → Mul → Reshape → Convert] ──┴─ Concat ─→ GatherMatmul
//
// After:   Concat(Const_1, Const_2) → [Convert → Sub(Concat(zp1,zp2)) →
//            Mul(Concat(s1,s2)) → Reshape → Convert] ─→ GatherMatmul
// ============================================================================

namespace {

// Recursively merge two parallel decompression chains by pushing Concat to leaf constants.
// For Reshape: the shape-target input (input 1) is adjusted to account for the doubled
//   concat-axis dimension, or kept as-is if it contains -1 (auto-adjusts).
// For all other nodes with matching types: all inputs are merged recursively.
// At Constants (leaf nodes): creates a folded Concat, or reuses one constant when
//   the concat axis exceeds the constant rank (broadcast-compatible constants like
//   scalar zero-points or scales).
// Returns empty Output on failure (mismatched topology).
ov::Output<ov::Node> merge_weight_branches(const ov::Output<ov::Node>& out1,
                                           const ov::Output<ov::Node>& out2,
                                           int64_t concat_axis,
                                           ov::NodeVector& new_nodes) {
    auto n1 = out1.get_node_shared_ptr();
    auto n2 = out2.get_node_shared_ptr();

    // If both sides point to the same node, just reuse it (shared constant/subgraph)
    if (n1 == n2) {
        return out1;
    }

    // Base case: both are Constants
    if (ov::is_type<v0::Constant>(n1) && ov::is_type<v0::Constant>(n2)) {
        auto c1 = ov::as_type_ptr<v0::Constant>(n1);
        auto c2 = ov::as_type_ptr<v0::Constant>(n2);

        const auto rank = static_cast<int64_t>(c1->get_shape().size());
        const auto pos_axis = concat_axis >= 0 ? concat_axis : rank + concat_axis;

        // If concat axis is out of range (e.g., scalar or lower-rank broadcast constants),
        // these constants apply identically to both branches via broadcasting.
        // Reuse one if shapes match; fail otherwise.
        if (pos_axis < 0 || pos_axis >= rank) {
            if (c1->get_shape() == c2->get_shape()) {
                return out1;
            }
            return {};
        }

        // Use raw byte-level concat for sub-byte types to avoid evaluation bugs
        auto merged = concat_constants_raw(c1, c2, concat_axis);
        if (!merged) {
            return {};
        }
        new_nodes.push_back(merged);
        return merged->output(0);
    }

    // Both must be the same op type with the same number of inputs
    if (n1->get_type_info() != n2->get_type_info()) {
        return {};
    }
    if (n1->get_input_size() != n2->get_input_size()) {
        return {};
    }

    // For Reshape: merge only the data input (0).  Adjust the target shape (input 1)
    // to account for the doubled concat-axis dimension if the target uses explicit values.
    if (ov::is_type<v1::Reshape>(n1)) {
        auto merged_data = merge_weight_branches(n1->input_value(0), n2->input_value(0), concat_axis, new_nodes);
        if (!merged_data.get_node()) {
            return {};
        }

        // Determine whether the target shape needs adjustment
        auto target1 = ov::as_type_ptr<v0::Constant>(n1->input_value(1).get_node_shared_ptr());
        auto target2 = ov::as_type_ptr<v0::Constant>(n2->input_value(1).get_node_shared_ptr());
        ov::Output<ov::Node> new_target = n1->input_value(1);

        if (target1 && target2) {
            auto vals1 = target1->cast_vector<int64_t>();
            auto vals2 = target2->cast_vector<int64_t>();
            // Check if any dimension uses -1 (auto-infer) — if so, it auto-adjusts
            bool has_auto = std::any_of(vals1.begin(), vals1.end(), [](int64_t v) { return v == -1; });
            if (!has_auto && vals1.size() == vals2.size()) {
                // All explicit values — adjust the concat-axis dimension
                const auto target_rank = static_cast<int64_t>(vals1.size());
                const auto pos_axis = concat_axis >= 0 ? concat_axis : target_rank + concat_axis;
                if (pos_axis >= 0 && pos_axis < target_rank) {
                    vals1[pos_axis] += vals2[pos_axis];
                    auto adjusted = v0::Constant::create(target1->get_element_type(), target1->get_shape(), vals1);
                    new_nodes.push_back(adjusted);
                    new_target = adjusted->output(0);
                }
            }
        }

        auto new_node = n1->clone_with_new_inputs({merged_data, new_target});
        new_nodes.push_back(new_node);
        return new_node->output(0);
    }

    // For all other node types (Convert, Subtract, Multiply, Transpose, etc.):
    // recursively merge every input from both branches.
    ov::OutputVector merged_inputs;
    for (size_t i = 0; i < n1->get_input_size(); ++i) {
        auto merged = merge_weight_branches(n1->input_value(i), n2->input_value(i), concat_axis, new_nodes);
        if (!merged.get_node()) {
            return {};
        }
        merged_inputs.push_back(merged);
    }

    auto new_node = n1->clone_with_new_inputs(merged_inputs);
    new_nodes.push_back(new_node);
    return new_node->output(0);
}

}  // namespace

FuseConcatIntoGatherMatmulWeights::FuseConcatIntoGatherMatmulWeights() {
    MATCHER_SCOPE(FuseConcatIntoGatherMatmulWeights);

    auto concat_m = pattern::wrap_type<v0::Concat>({pattern::any_input(), pattern::any_input()});
    auto bgm_m = pattern::wrap_type<ov::op::internal::GatherMatmul>(
        {pattern::any_input(), concat_m, pattern::any_input(), pattern::any_input()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        auto bgm_node = pm.at(bgm_m).get_node_shared_ptr();
        auto concat_node = ov::as_type_ptr<v0::Concat>(pm.at(concat_m).get_node_shared_ptr());
        if (!concat_node || concat_node->get_input_size() != 2) {
            return false;
        }

        auto branch1 = concat_node->input_value(0);
        auto branch2 = concat_node->input_value(1);

        // If the Concat is already folded to a Constant, nothing to do
        if (ov::is_type<v0::Constant>(concat_node)) {
            return false;
        }

        int64_t concat_axis = concat_node->get_axis();

        ov::NodeVector new_nodes;
        auto merged = merge_weight_branches(branch1, branch2, concat_axis, new_nodes);
        if (!merged.get_node()) {
            return false;
        }

        // Validate the merged subgraph
        for (const auto& node : new_nodes) {
            node->validate_and_infer_types();
        }

        // Replace the Concat output with the merged chain output
        concat_node->output(0).replace(merged);
        ov::copy_runtime_info(concat_node, new_nodes);

        // Re-validate GatherMatmul with the new B input
        bgm_node->validate_and_infer_types();

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(bgm_m, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
