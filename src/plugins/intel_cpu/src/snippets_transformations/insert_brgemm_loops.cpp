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
InsertBrgemmLoops::InsertBrgemmLoops(size_t M_block_size) : ModelPass(), m_M_block_size{M_block_size} {
}

bool InsertBrgemmLoops::run_on_model(const std::shared_ptr<ov::Model>& m) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::InsertBrgemmLoops")
    RUN_ON_MODEL_SCOPE(InsertBrgemmLoops);
    const auto& ordered_ops = m->get_ordered_ops();
    bool modified = false;
    for (size_t i = 0; i < ordered_ops.size(); i++) {
        const auto& node = ordered_ops[i];
        if (const auto& brgemm = as_type_ptr<ngraph::snippets::op::Brgemm>(node)) {
            const auto& shape_A = ngraph::snippets::utils::get_port_planar_shape(brgemm->input_value(0));
            const auto M_rows = shape_A[shape_A.size() - 2].get_length();
            // Note: We always insert Loop around Brgemm even if there is a single iteration.
            // In this case the loop is needed to apply finalization offsets if the Brgemm is followed by a buffer.
            const auto& loop_begin = ngraph::snippets::op::insertLoopBegin(brgemm->input_values());
            std::vector<int64_t> ptr_increments(brgemm->get_input_size() + brgemm->get_output_size(), 0);
            ptr_increments.front() = static_cast<int64_t>(m_M_block_size * brgemm->get_leading_dim_a());
            ptr_increments.back() = static_cast<int64_t>(m_M_block_size * brgemm->get_leading_dim_c());
            std::vector<int64_t> finalization_offsets(ptr_increments.size(), 0);
            // Note: need to reset output data pointer if is connected to a Buffer
            for (const auto& consumer_in : brgemm->get_output_target_inputs(0)) {
                if (ov::is_type<ngraph::snippets::op::Buffer>(consumer_in.get_node()->shared_from_this())) {
                    finalization_offsets.back() = -static_cast<int64_t>(brgemm->get_leading_dim_c() * M_rows);
                    break;
                }
            }

            std::shared_ptr<ov::Node> last_node_in_loop = brgemm;
            for (size_t j = i; j < ordered_ops.size(); j++) {
                const auto& n = ordered_ops[j];
                if (const auto& buffer = as_type_ptr<ngraph::snippets::op::Buffer>(n)) {
                    buffer->set_allocation_shape({m_M_block_size, brgemm->get_leading_dim_c()});
                    last_node_in_loop = n;
                } else if (const auto& loop_begin = as_type_ptr<ngraph::snippets::op::LoopBegin>(n)) {
                    auto loop_end = loop_begin->get_loop_end();
                    // todo: Do we need make this LoopBegin inputs to be Brgemm LoopBegin inputs as well?
                    //  in a general case it's needed to guarantee correct operation ordering. But do we need it for our patterns?
                    for (const auto& input : loop_begin->inputs()) {
                        if (is_type<ngraph::snippets::op::Buffer>(input.get_source_output().get_node_shared_ptr())) {
                            auto inner_fin_offsets = loop_end->get_finalization_offsets();
                            inner_fin_offsets[input.get_index()] = -ptr_increments.back();
                            loop_end->set_finalization_offsets(inner_fin_offsets);
                            ptr_increments.back() = 0;
                            finalization_offsets.back() = 0;
                        }
                    }
                    loop_end->set_work_amount(m_M_block_size);
                    while (ordered_ops[j] != loop_end)
                        j++;
                    last_node_in_loop = ordered_ops[j];
                } else if (is_type<ngraph::snippets::op::Brgemm>(n)) {
                    last_node_in_loop = n;
                } else {
                    break;
                }
            }

            std::vector<Input<Node>> before_these_inputs;
            for (const auto& out : last_node_in_loop->outputs()) {
                for (const auto& in : out.get_target_inputs())
                    before_these_inputs.push_back(in);
            }
            insertLoopEnd(before_these_inputs, loop_begin, M_rows, m_M_block_size,
                          ptr_increments, finalization_offsets);
            brgemm->set_input_count(m_M_block_size, 0);
            modified = true;
        }
    }
    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov