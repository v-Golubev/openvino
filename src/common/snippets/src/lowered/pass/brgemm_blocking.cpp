// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/brgemm_blocking.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool BrgemmBlockingBase::blocking_loop_exists(const snippets::lowered::LoopManagerPtr& loop_manager,
                                              const ExpressionPtr& brgemm_expr,
                                              const std::shared_ptr<snippets::op::Brgemm>& brgemm) {
    auto check_port = [&](const LoopPort& p) {
        return p.expr_port->get_expr() == brgemm_expr && ov::snippets::utils::one_of(p.dim_idx, 0ul, 1ul);
    };

    const auto& loop_ids = brgemm_expr->get_loop_ids();
    for (const auto& id : loop_ids) {
        const auto loop = loop_manager->get_loop_info(id);
        if (std::any_of(loop->get_input_ports().begin(), loop->get_input_ports().end(), check_port) ||
            std::any_of(loop->get_output_ports().begin(), loop->get_output_ports().end(), check_port)) {
            return true;
        }
    }
    return false;
}

bool BrgemmBlockingBase::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmCPUBlocking")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool modified = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto& node = brgemm_expr->get_node();
        const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(node);
        if (!brgemm || blocking_loop_exists(loop_manager, brgemm_expr, brgemm))
            continue;
        modified = mark_blocking_loops(linear_ir, expr_it);
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov