// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = LinearIR::LoopManager::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

BrgemmBlocking::BrgemmBlocking() : Pass() {}

bool BrgemmBlocking::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    auto blocking_loop_exists = [&](const ExpressionPtr& brgemm_expr, const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        auto check_port = [&](const LoopPort& p) {
            return p.expr_port->get_expr() == brgemm_expr && (p.dim_idx == 0 || p.dim_idx == 1);
        };

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (std::any_of(loop->entry_points.begin(), loop->entry_points.end(), check_port) ||
                std::any_of(loop->exit_points.begin(), loop->exit_points.end(), check_port)) {
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

        const auto& input_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& input_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& output_desc = brgemm_expr->get_output_port_descriptor(0);

        auto input_0_subtensor = input_0_desc->get_subtensor();
        auto input_1_subtensor = input_1_desc->get_subtensor();
        auto output_subtensor = output_desc->get_subtensor();

        auto apply_m_blocking = [&]() {
            const auto& input_shape_0 = input_0_desc->get_shape();
            const auto& input_layout_0 = input_0_desc->get_layout();

            const auto& m_idx = *(input_layout_0.rbegin() + 1);
            const auto& m = input_shape_0[m_idx];
            const auto block_size_m = brgemm->get_m_block_size();
            if (block_size_m >= m) {
                *(input_0_subtensor.rbegin() + 1) = m;
                *(output_subtensor.rbegin() + 1) = m;
            } else {
                *(input_0_subtensor.rbegin() + 1) = block_size_m;
                *(output_subtensor.rbegin() + 1) = block_size_m;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(brgemm_expr->get_input_port(1), false)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(expr_it, std::next(expr_it), m, block_size_m, 1, entries, exits);
            }
        };

        auto apply_n_blocking = [&]() {
            const auto& input_shape_1 = input_1_desc->get_shape();
            const auto& input_layout_1 = input_1_desc->get_layout();

            const auto& n_idx = *input_layout_1.rbegin();
            const auto& n = input_shape_1[n_idx];
            const auto block_size_n = brgemm->get_n_block_size();
            if (block_size_n >= n) {
                *input_1_subtensor.rbegin() = n;
                *output_subtensor.rbegin() = n;
            } else {
                *input_1_subtensor.rbegin() = block_size_n;
                *output_subtensor.rbegin() = block_size_n;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                              LoopPort(brgemm_expr->get_input_port(1), true)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(expr_it, std::next(expr_it), n, block_size_n, 0, entries, exits);
            }
        };

        auto apply_k_blocking = [&]() {
            const auto& input_shape_0 = input_0_desc->get_shape();
            const auto& input_layout_0 = input_0_desc->get_layout();

            const auto& k_idx = *input_layout_0.rbegin();
            const auto& k = input_shape_0[k_idx];
            const auto block_size_k = brgemm->get_k_block_size();
            if (block_size_k >= k) {
                *input_0_subtensor.rbegin() = k;
                *(input_1_subtensor.rbegin() + 1) = k;
            } else {
                *input_0_subtensor.rbegin() = block_size_k;
                *(input_1_subtensor.rbegin() + 1) = block_size_k;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                              LoopPort(brgemm_expr->get_input_port(1), true, 1)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), true, 1);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
                auto loop_id = loop_manager->mark_loop(expr_it, std::next(expr_it), k, block_size_k, entries, exits);
                loop_manager->get_loop_info(loop_id)->brgemm_k_blocking_loop = true;
            }
        };

        apply_k_blocking();
        apply_n_blocking();
        apply_m_blocking();

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor(input_0_subtensor);
        brgemm_expr->get_input_port_descriptor(1)->set_subtensor(input_1_subtensor);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor(output_subtensor);

        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov