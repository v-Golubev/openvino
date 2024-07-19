// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu_blocking.hpp"

#include "cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::snippets::lowered;

LinearIR::constExprIt BrgemmCPUBlocking::move_new_memory_buffer(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
    return std::prev(brgemm_it);
}

LinearIR::constExprIt BrgemmCPUBlocking::get_loop_begin_pos(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it, const ExpressionPtr& copy_b_expr) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto node = brgemm_expr->get_node();
    const auto brgemm = ov::as_type_ptr<intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "get_loop_begin_pos must be called only for BrgemmCPU expression");
    if (brgemm && brgemm->is_amx()) {
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    }
    if (copy_b_expr) {
        loop_begin_it = linear_ir.find(copy_b_expr);
    }
    return loop_begin_it;
}

bool BrgemmCPUBlocking::mark_blocking_loops(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    if (!brgemm)
        return false;

    const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
    const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
    const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

    const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
    const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
    const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

    auto in_0_subtensor = in_0_desc->get_subtensor();
    auto in_1_subtensor = in_1_desc->get_subtensor();
    auto out_subtensor = out_desc->get_subtensor();

    const auto& m = *++out_preordered_dims.rbegin();
    const auto& n = *out_preordered_dims.rbegin();
    const auto& k = *in_0_planar_dims.rbegin();
    OPENVINO_ASSERT(k == *++in_1_planar_dims.rbegin(), "Brgemm input descriptors have different K dimension value.");

    const auto block_size_m = snippets::utils::is_dynamic_value(m) ? brgemm->get_m_block_size() : std::min(brgemm->get_m_block_size(), m);
    const auto block_size_n = snippets::utils::is_dynamic_value(n) ? brgemm->get_n_block_size() : std::min(brgemm->get_n_block_size(), n);
    const auto block_size_k = snippets::utils::is_dynamic_value(k) ? brgemm->get_k_block_size() : std::min(brgemm->get_k_block_size(), k);

    brgemm_expr->get_input_port_descriptor(0)->set_subtensor(ov::snippets::VectorDims{block_size_m, block_size_k});
    brgemm_expr->get_input_port_descriptor(1)->set_subtensor(ov::snippets::VectorDims{block_size_k, block_size_n});
    brgemm_expr->get_output_port_descriptor(0)->set_subtensor(ov::snippets::VectorDims{block_size_m, block_size_n});

    ov::snippets::lowered::ExpressionPtr copy_b_expr = nullptr;
    if (brgemm && brgemm->is_with_data_repacking()) {
        const auto copy_b = brgemm->get_brgemm_copy();
        const auto copy_b_k_block = snippets::utils::is_dynamic_value(k) ? copy_b->get_k_block_size() : std::min(copy_b->get_k_block_size(), k);
        const auto copy_b_n_block = snippets::utils::is_dynamic_value(n) ? copy_b->get_n_block_size() : std::min(copy_b->get_n_block_size(), n);
        OPENVINO_ASSERT(snippets::utils::one_of(copy_b_k_block, k, block_size_k),
                        "CopyB has unexpected K block size (", copy_b->get_k_block_size(),
                        "). It must be equal to K dim (", k, ")",
                        " or to brgemm's k block size (", block_size_k, ")");
        OPENVINO_ASSERT(snippets::utils::one_of(copy_b_n_block, n, block_size_n),
                        "CopyB has unexpected n block size (", copy_b->get_n_block_size(),
                        "). It must be equal to n dim (", n, ")",
                        " or to brgemm's n block size (", block_size_n, ")");

        const auto& repacking_expr = linear_ir.get_expr_by_node(copy_b);
        const ov::snippets::VectorDims repacking_subtensor{copy_b_k_block, copy_b_n_block};
        repacking_expr->get_input_port_descriptor(0)->set_subtensor(repacking_subtensor);
        repacking_expr->get_output_port_descriptor(0)->set_subtensor(repacking_subtensor);
        if (copy_b->is_with_compensations()) {
            const ov::snippets::VectorDims compensations_subtensor{1, copy_b_n_block};
            OPENVINO_ASSERT(brgemm_expr->get_input_count() == 3, "Brgemm must have 3 inputs in case of compensations.");
            brgemm_expr->get_input_port_descriptor(2)->set_subtensor(compensations_subtensor);
            repacking_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
        }

        // If copyB block sizes are equal to brgemm ones, copy_b_expr is covered by blocking loops as well
        if (copy_b_k_block == block_size_k && copy_b_n_block == block_size_n)
            copy_b_expr = repacking_expr;
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    auto mark_m_blocking = [&](bool include_repacking) {
        const auto loop_begin_it = get_loop_begin_pos(linear_ir, brgemm_it, include_repacking ? copy_b_expr : nullptr);
        const auto loop_end_it = std::next(brgemm_it);

        const auto b_input_port = include_repacking && copy_b_expr ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1);
        std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(b_input_port, false)};
        if (!include_repacking && brgemm && brgemm->is_with_compensations())
            entries.emplace_back(brgemm_expr->get_input_port(2), false);
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
    };

    auto mark_n_blocking = [&]() {
        const auto loop_begin_it = get_loop_begin_pos(linear_ir, brgemm_it, copy_b_expr);
        const auto loop_end_it = std::next(brgemm_it);

        const std::vector<LoopPort> entries{
            LoopPort(brgemm_expr->get_input_port(0), false),
            LoopPort(copy_b_expr ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
    };

    auto mark_k_blocking = [&]() {
        const auto loop_begin_it = get_loop_begin_pos(linear_ir, brgemm_it, copy_b_expr);
        const auto loop_end_it = std::next(brgemm_it);

        const std::vector<LoopPort> entries{
            LoopPort(brgemm_expr->get_input_port(0), true, 0),
            LoopPort(copy_b_expr ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true, 1)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
        const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
        const auto& loop_info = loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id);
        loop_info->register_pass_to_handler<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER, SetBrgemmBeta>(0.f);
    };

    const bool k_blocking = block_size_k != k;
    const bool n_blocking = block_size_n != n;
    const bool m_blocking = block_size_m != m;
    // It is not necessary to include copyB in loop by M if there are no blocking by KN
    const bool include_repacking_in_loop = k_blocking || n_blocking;

    if (k_blocking) {
        mark_k_blocking();
    } else {
        brgemm->set_beta(0.f);
    }
    if (n_blocking)
        mark_n_blocking();
    if (m_blocking)
        mark_m_blocking(include_repacking_in_loop);
    return true;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov