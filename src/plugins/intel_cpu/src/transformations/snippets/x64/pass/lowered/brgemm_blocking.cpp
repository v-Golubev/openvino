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
using LoopManager = snippets::lowered::LinearIR::LoopManager;
using LoopInfoPtr = LoopManager::LoopInfoPtr;
using LoopPort = LoopManager::LoopPort;

BrgemmBlocking::BrgemmBlocking() : Pass() {}

bool BrgemmBlocking::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;


    const auto& loop_manager = linear_ir.get_loop_manager();
    const size_t dim_idx_m = 1;
    const size_t dim_idx_n = 0;

    auto blocking_loop_exists = [&](const ov::snippets::lowered::ExpressionPtr& expr,
                                    const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        const auto& loop_ids = expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (loop->dim_idx == dim_idx_m) {
                OPENVINO_ASSERT(brgemm->get_input_count(0) == loop->increment,
                                "Brgemm ", brgemm, " has input count (", brgemm->get_input_count(0),
                                ") which doesn't match the increment(", loop->increment, ") of loop by M");
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
        if (!brgemm || blocking_loop_exists(expr, brgemm))
            continue;

        const auto& input_0_desc = expr->get_input_port_descriptor(0);
        const auto& input_1_desc = expr->get_input_port_descriptor(1);
        const auto& output_desc = expr->get_output_port_descriptor(0);

        auto input_0_subtensor = input_0_desc->get_subtensor();
        auto input_1_subtensor = input_1_desc->get_subtensor();
        auto output_subtensor = output_desc->get_subtensor();

        auto apply_m_blocking = [&]() {
            const auto& input_shape_0 = input_0_desc->get_shape();
            const auto& input_layout_0 = input_0_desc->get_layout();

            const auto& m_idx = *(input_layout_0.rbegin() + dim_idx_m);
            const auto& m = input_shape_0[m_idx];
            const auto block_size_m = brgemm->get_m_block_size();
            *(input_0_subtensor.rbegin() + 1) = block_size_m;
            *(output_subtensor.rbegin() + 1) = block_size_m;

            std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), true), LoopPort(expr->get_input_port(1), false)};
            if (brgemm->is_with_scratchpad())
                entries.emplace_back(expr->get_input_port(2), false);
            std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
            loop_manager->mark_loop(expr_it, std::next(expr_it), m, block_size_m, dim_idx_m, entries, exits);
        };

        auto apply_n_blocking = [&]() {
            const auto& input_shape_1 = input_1_desc->get_shape();
            const auto& input_layout_1 = input_1_desc->get_layout();

            const auto& n_idx = *(input_layout_1.rbegin() + dim_idx_n);
            const auto& n = input_shape_1[n_idx];
            const auto block_size_n = brgemm->get_n_block_size();
            *input_1_subtensor.rbegin() = block_size_n;
            *output_subtensor.rbegin() = block_size_n;

            std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), false), LoopPort(expr->get_input_port(1), true)};
            if (brgemm->is_with_scratchpad())
                entries.emplace_back(expr->get_input_port(2), true);
            std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
            loop_manager->mark_loop(expr_it, std::next(expr_it), n, block_size_n, dim_idx_n, entries, exits);
        };

        apply_m_blocking();
        apply_n_blocking();

        input_0_desc->set_subtensor(input_0_subtensor);
        input_1_desc->set_subtensor(input_1_subtensor);
        output_desc->set_subtensor(output_subtensor);
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov