// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_tpp_blocking.hpp"

#include "transformations/snippets/x64/pass/lowered/cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::snippets::lowered;

bool BrgemmTPPBlocking::mark_blocking_loops(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(brgemm_expr->get_node());
    if (!brgemm)
        return false;
    const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
    const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
    const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

    const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
    const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
    const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

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

    const auto& loop_manager = linear_ir.get_loop_manager();
    auto mark_m_blocking = [&]() {
        const auto loop_begin_it = brgemm_it;
        const auto loop_end_it = std::next(brgemm_it);
        std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(brgemm_expr->get_input_port(1), false)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
    };

    auto mark_n_blocking = [&]() {
        const auto loop_begin_it = brgemm_it;
        const auto loop_end_it = std::next(brgemm_it);

        const std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                            LoopPort(brgemm_expr->get_input_port(1), true)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
    };

    auto mark_k_blocking = [&]() {
        const auto loop_begin_it = brgemm_it;
        const auto loop_end_it = std::next(brgemm_it);

        const std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                            LoopPort(brgemm_expr->get_input_port(1), true, 1)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
        const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
        const auto& loop_info = loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id);
        loop_info->register_pass_to_handler<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER,
                                            ov::intel_cpu::pass::SetBrgemmBeta>(0.f);
    };

    const bool k_blocking = block_size_k != k;
    const bool n_blocking = block_size_n != n;
    const bool m_blocking = block_size_m != m;
    if (k_blocking) {
        mark_k_blocking();
    } else {
        brgemm->set_beta(0.f);
    }
    if (n_blocking)
        mark_n_blocking();
    if (m_blocking)
        mark_m_blocking();
    return true;
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
