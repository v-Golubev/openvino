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

InsertBrgemmLoops::InsertBrgemmLoops(size_t M_block_size) : ModelPass(), m_M_block_size{M_block_size} {}

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
            brgemm->set_input_count(m_M_block_size, 0);

            // Note: We always insert Loop around Brgemm even if there is a single iteration.
            // In this case the loop is needed to apply finalization offsets if the Brgemm is followed by a buffer.
            std::vector<ov::Input<ov::Node>> loop_begin_inputs = brgemm->inputs();
            ov::NodeVector body_ops = { brgemm };

            size_t j;
            for (j = i + 1; j < ordered_ops.size(); j++) {
                const auto& n = ordered_ops[j];
                if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::Buffer>(n)) {
                    buffer->set_allocation_shape({m_M_block_size, brgemm->get_leading_dim_c()});
                    body_ops.push_back(n);
                    continue;
                }

                if (const auto& loop_begin = as_type_ptr<ngraph::snippets::op::LoopBegin>(n)) {
                    const auto& loop_end = loop_begin->get_loop_end();
                    loop_end->set_work_amount(m_M_block_size);

                    // Unite Loop inputs
                    const auto& local_loop_begin_inputs = loop_begin->inputs();
                    for (const auto& input : local_loop_begin_inputs) {
                        const auto& parent = input.get_source_output().get_node_shared_ptr();
                        if (std::find(body_ops.begin(), body_ops.end(), parent) == body_ops.end()) {
                            loop_begin_inputs.push_back(input);
                        }
                        if (const auto buffer = as_type_ptr<ngraph::snippets::op::Buffer>(parent)) {
                            auto inner_fin_offsets = loop_end->get_finalization_offsets();
                            inner_fin_offsets[input.get_index()] = -1 * static_cast<int64_t>(ngraph::shape_size(buffer->get_allocation_shape()));
                            loop_end->set_finalization_offsets(inner_fin_offsets);
                        }
                    }

                    while (ordered_ops[j] != loop_end) {
                        if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::Buffer>(ordered_ops[j])) {
                            buffer->set_allocation_shape({m_M_block_size, brgemm->get_leading_dim_c()});
                        }
                        body_ops.push_back(ordered_ops[j]);
                        j++;
                    }
                    body_ops.push_back(loop_end);
                    continue;
                }

                if (const auto current_brgemm = ov::as_type_ptr<ngraph::snippets::op::Brgemm>(n)) {
                    // If the parent of first input of Brgemm is in Loop, we fuse Brgemm into the Loop
                    // to carry on blocking by M of first port
                    // Otherwise, we cannot fuse Brgemm to Loop - break
                    if (std::find(ordered_ops.begin() + i, ordered_ops.begin() + j, n->get_input_node_shared_ptr(0)) == ordered_ops.begin() + j) {
                        break;
                    }

                    current_brgemm->set_input_count(m_M_block_size, 0);
                    loop_begin_inputs.push_back(n->input(1));
                    body_ops.push_back(n);
                    continue;
                }

                break;
            }

            ov::OutputVector loop_begin_input_sources;
            std::vector<int64_t> ptr_increments, finalization_offsets;
            for (const auto& input : loop_begin_inputs) {
                loop_begin_input_sources.push_back(input.get_source_output());
                const auto current_brgemm = ov::as_type_ptr<ngraph::snippets::op::Brgemm>(input.get_node()->shared_from_this());
                if (current_brgemm && input.get_index() == 0) {
                    ptr_increments.push_back(static_cast<int64_t>(m_M_block_size * current_brgemm->get_leading_dim_a()));
                } else {
                    ptr_increments.push_back(0);
                }
            }

            std::vector<Input<Node>> before_these_inputs;
            const auto& last_node = body_ops.back();
            for (const auto& out : last_node->outputs()) {
                for (const auto& in : out.get_target_inputs()) {
                    before_these_inputs.push_back(in);
                    if (const auto current_brgemm = ov::as_type_ptr<ngraph::snippets::op::Brgemm>(last_node)) {
                        ptr_increments.push_back(static_cast<int64_t>(m_M_block_size * current_brgemm->get_leading_dim_c()));
                    } else {
                        ptr_increments.push_back(0);
                    }
                }
            }

            const auto& loop_begin = ngraph::snippets::op::insertLoopBegin(loop_begin_input_sources);
            insertLoopEnd(before_these_inputs, loop_begin, M_rows, m_M_block_size, ptr_increments, finalization_offsets);
            modified = true;

            // skip the current new Loop
            i = j;
        }
    }
    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
