// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "insert_brgemm_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "op/brgemm_cpu.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "snippets/utils.hpp"

#include <ngraph/rt_info.hpp>
namespace ov {
namespace intel_cpu {
namespace pass {


InsertBrgemmLoops::InsertBrgemmLoops(size_t M_block_size) {
    MATCHER_SCOPE(InsertBrgemmLoops);
    auto brgemm_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Brgemm>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::InsertBrgemmLoops")
        const auto& brgemm = as_type_ptr<ngraph::snippets::op::Brgemm>(m.get_match_root());
        const auto& shape_A =   ngraph::snippets::utils::get_port_planar_shape(brgemm->input_value(0));
        const auto M_rows = shape_A[shape_A.size() - 2].get_length();
        // Note: We always insert Loop around Brgemm even if there is a single iteration.
        // In this case the loop is needed to apply finalization offsets if the Brgemm is followed by a buffer.
        const auto& loop_begin = ngraph::snippets::op::insertLoopBegin(brgemm->input_values());
        const std::vector<int64_t> ptr_increments {static_cast<int64_t>(M_block_size *  brgemm->get_leading_dim_a()),
                                                   0,
                                                   static_cast<int64_t>(M_block_size * brgemm->get_leading_dim_c())};
        std::vector<int64_t> finalization_offsets(ptr_increments.size(), 0);
        // Note: need to reset output data pointer if is connected to a Buffer
        for (const auto& consumer_in : brgemm->get_output_target_inputs(0)) {
            if (ov::is_type<ngraph::snippets::op::Buffer>(consumer_in.get_node()->shared_from_this())) {
                finalization_offsets.back() = -static_cast<int64_t>(brgemm->get_leading_dim_c() * M_rows);
                break;
            }
        }

        std::vector<Input<Node>> child_inputs;
        for (const auto& in : brgemm->output(0).get_target_inputs())
            child_inputs.push_back(in);
        insertLoopEnd(child_inputs, loop_begin, M_rows, M_block_size,
                      ptr_increments,  finalization_offsets);
        brgemm->set_input_count(M_block_size, 0);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(brgemm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov