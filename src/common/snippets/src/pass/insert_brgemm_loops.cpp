// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/insert_brgemm_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "snippets/utils.hpp"

#include <ngraph/rt_info.hpp>
namespace ngraph {
namespace snippets {
namespace pass {
namespace {
std::pair<std::vector<size_t>, size_t> get_node_layout_and_leading_dimension(const Output<Node>& in) {
    auto in_node = in.get_node_shared_ptr();
    // If input is LoopBegin then it has multiple outputs and doesn't store output layout,
    // so we have to check the original input node rt_info
    if (ov::is_type<snippets::op::LoopBegin>(in_node)) {
        in_node = in_node->get_input_node_shared_ptr(in.get_index());;
    }
    auto layout = ngraph::snippets::utils::get_node_output_layout(in_node);
    size_t leading_dimension;
    const auto& io_shape = in.get_shape();
    if (layout.empty()) {
        // empty value indicates a planar layout
        leading_dimension = io_shape.back();
        layout.resize(io_shape.size());
        std::iota(layout.begin(), layout.end(), 0);
    } else {
        // The idea here is to find "2" (for 4D shapes) in the layout and multiply dimensions that are to the right
        // This implies that "3" is the last layout value, otherwise this layout is not supported.
        // counting from the end since shape could be prepended with ones
        const int64_t num_last_dims = layout.end() - std::find(layout.begin(), layout.end(), layout.size() - 2) - 1;
        if (layout.back() != layout.size() - 1 || num_last_dims < 1)
            throw ngraph_error("Brgemm detected unschedulable shape + layout combination");
        leading_dimension = std::accumulate(io_shape.end() - num_last_dims, io_shape.end(), 1, std::multiplies<size_t>());
    }
    return {layout, leading_dimension};
}
} //namespace


InsertBrgemmLoops::InsertBrgemmLoops() {
    MATCHER_SCOPE(InsertBrgemmLoops);
    auto brgemm_pattern = pattern::wrap_type<snippets::op::Brgemm>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::InsertBrgemmLoops")
        const auto& brgemm = as_type_ptr<snippets::op::Brgemm>(m.get_match_root());
        const auto M_block_size = brgemm->get_M_block_size();
        const auto& shape_A =   utils::get_port_planar_shape(brgemm->input_value(0));
        const auto M_rows = shape_A[shape_A.size() - 2].get_length();
        if (M_rows > M_block_size) {
            const auto& loop_begin = op::insertLoopBegin(brgemm->input_values());
            const auto leading_dim_C = get_node_layout_and_leading_dimension(brgemm->output(0)).second;
            const auto leading_dim_A = get_node_layout_and_leading_dimension(brgemm->input_value(0)).second;
            const std::vector<int64_t> ptr_increments {static_cast<int64_t>(M_block_size * leading_dim_A),
                                                       0,
                                                       static_cast<int64_t>(M_block_size * leading_dim_C)};
            const std::vector<int64_t> finalization_offsets(ptr_increments.size(), 0);

            std::vector<Input<Node>> child_inputs;
            for (const auto& in : brgemm->output(0).get_target_inputs())
                child_inputs.push_back(in);
            insertLoopEnd(child_inputs, loop_begin, M_rows, M_block_size,
                          ptr_increments,  finalization_offsets);
            return true;
        }
        brgemm->set_input_count(M_rows, 0);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(brgemm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph