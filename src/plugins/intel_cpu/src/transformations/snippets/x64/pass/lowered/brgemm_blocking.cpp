// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/insert_tail_loop.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = LinearIR::LoopManager::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using LoopInfo = LinearIR::LoopManager::LoopInfo;
using namespace ov::snippets::lowered::pass;

BrgemmBlocking::BrgemmBlocking() : Pass() {}

void BrgemmBlocking::move_amx_scratchpad_buffer(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
}

bool BrgemmBlocking::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    auto blocking_loop_exists = [&](const ExpressionPtr& brgemm_expr, const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        auto check_port = [&](const LoopPort& p) {
            return p.expr_port->get_expr() == brgemm_expr && ov::snippets::utils::one_of(p.dim_idx, 0ul, 1ul);
        };

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (std::any_of(loop->get_entry_points().begin(), loop->get_entry_points().end(), check_port) ||
                std::any_of(loop->get_exit_points().begin(), loop->get_exit_points().end(), check_port)) {
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
        if (!brgemm || blocking_loop_exists(brgemm_expr, brgemm))
            continue;

        const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

        const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
        const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
        const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

        auto apply_m_blocking = [&]() {
            const auto& m = *(out_preordered_dims.rbegin() + 1);
            const auto block_size_m = brgemm->get_m_block_size();
            if (block_size_m >= m) {
                brgemm->set_m_block_size(m);
            } else {
                brgemm->set_m_block_size(block_size_m);
                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true),
                                              LoopPort(brgemm_expr->get_input_port(1), false)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
                } else if (brgemm->is_amx()) {
                    move_amx_scratchpad_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
                const auto tail_size = m % block_size_m;
                if (tail_size != 0) {
                    const auto& loop_info = loop_manager->get_loop_info(id);
                    loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ReduceWorkAmount>(tail_size);
                    loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ZeroFinalizationOffsets>();
                    loop_info->handlers[LoopInfo::LAST_ITER].register_pass<DefaultTailLoopHandler>(tail_size);
                    loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmMBlockSize>(tail_size);
                }
            }
        };

        auto apply_n_blocking = [&]() {
            const auto& n = *out_preordered_dims.rbegin();
            const auto block_size_n = brgemm->get_n_block_size();
            if (block_size_n >= n) {
                brgemm->set_n_block_size(n);
            } else {
                brgemm->set_n_block_size(block_size_n);
                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                              LoopPort(brgemm_expr->get_input_port(1), true)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
                } else if (brgemm->is_amx()) {
                    move_amx_scratchpad_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
                const auto tail_size = n % block_size_n;
                if (tail_size != 0) {
                    const auto& loop_info = loop_manager->get_loop_info(id);
                    loop_info->handlers[LoopInfo::LAST_ITER].register_pass<DefaultTailLoopHandler>(tail_size);
                    loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmNBlockSize>(tail_size);
                    loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ReduceWorkAmount>(tail_size);
                    loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ZeroFinalizationOffsets>();
                }
            }
        };

        auto apply_k_blocking = [&]() {
            const auto& k = *in_0_planar_dims.rbegin();
            OPENVINO_ASSERT(k == *(in_1_planar_dims.rbegin() + 1), "Brgemm input descriptors have different K dimension value.");
            const auto block_size_k = brgemm->get_k_block_size();
            if (block_size_k >= k) {
                brgemm->set_k_block_size(k);
            } else {
                brgemm->set_k_block_size(block_size_k);
                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                              LoopPort(brgemm_expr->get_input_port(1), true, 1)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false, 1);
                } else if (brgemm->is_amx()) {
                    move_amx_scratchpad_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
                const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
                const auto loop_info = loop_manager->get_loop_info(id);
                const auto tail_size = k % block_size_k;
                if (tail_size != 0) {
                    if (k <= 2 * block_size_k) {
                        // First iter and tail loop
                        loop_info->handlers[LoopInfo::FIRST_ITER].register_pass<SetSingleIterationWithWorkAmount>(block_size_k);
                        loop_info->handlers[LoopInfo::FIRST_ITER].register_pass<ZeroFinalizationOffsets>();
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<DefaultTailLoopHandler>(tail_size);
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmKBlockSize>(tail_size);
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmBeta>(1.f);
                    } else {
                        // First iter, main body and tail loop
                        loop_info->handlers[LoopInfo::FIRST_ITER].register_pass<SetSingleIterationWithWorkAmount>(block_size_k);
                        loop_info->handlers[LoopInfo::FIRST_ITER].register_pass<ZeroFinalizationOffsets>();
                        loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ReduceWorkAmount>(block_size_k + tail_size);
                        loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<ZeroFinalizationOffsets>();
                        loop_info->handlers[LoopInfo::MAIN_BODY].register_pass<SetBrgemmBeta>(1.f);
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<DefaultTailLoopHandler>(tail_size);
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmKBlockSize>(tail_size);
                        loop_info->handlers[LoopInfo::LAST_ITER].register_pass<SetBrgemmBeta>(1.f);
                    }
                }
            }
        };

        apply_k_blocking();
        apply_n_blocking();
        apply_m_blocking();
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov