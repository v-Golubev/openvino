// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu_blocking.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;

bool BrgemmCPUBlocking::DummyPass::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    return true;
}
std::shared_ptr<snippets::lowered::pass::PassBase> BrgemmCPUBlocking::DummyPass::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

LinearIR::constExprIt BrgemmCPUBlocking::move_new_memory_buffer(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(wsp_expr);
    OPENVINO_ASSERT(wsp_buffer && wsp_buffer->is_independent_memory(), "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
    return std::prev(brgemm_it);
}

LinearIR::constExprIt BrgemmCPUBlocking::move_brgemm_copy_a(LinearIR& linear_ir, const LinearIR::constExprIt& insert_it,
                                                            const LinearIR::constExprIt& brgemm_copy_a_it) {
    if (*brgemm_copy_a_it != *std::prev(insert_it)) {
        linear_ir.move(brgemm_copy_a_it, insert_it);
    }
    return std::prev(insert_it);
}

LinearIR::constExprIt BrgemmCPUBlocking::get_loop_begin_pos(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it,
                                                            const ExpressionPtr& copy_a_expr, const ExpressionPtr& copy_b_expr) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "get_loop_begin_pos must be called only for BrgemmCPU expression");
    if (brgemm->get_config().is_amx())
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    if (copy_b_expr)
        loop_begin_it = linear_ir.find(copy_b_expr);
    if (copy_a_expr)
        loop_begin_it = move_brgemm_copy_a(linear_ir, loop_begin_it, linear_ir.find(copy_a_expr));
    return loop_begin_it;
}

size_t BrgemmCPUBlocking::get_default_n_blk(size_t n) const {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
}

SpecificIterationHandlers BrgemmCPUBlocking::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    SpecificIterationHandlers handlers = ov::snippets::lowered::pass::BrgemmBlockingBase::get_k_loop_handlers(work_amount, block_size);
    handlers.register_pass<SpecificLoopIterType::FIRST_ITER, DummyPass>();
    return handlers;
}

bool BrgemmCPUBlocking::mark_blocking_loops(LinearIR& linear_ir,
                                            const LinearIR::constExprIt& brgemm_it,
                                            size_t m_block,
                                            size_t n_block,
                                            size_t k_block) {
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    const auto config = brgemm->get_config();

    auto res = ov::snippets::lowered::pass::BrgemmBlockingBase::mark_blocking_loops(linear_ir, brgemm_it, m_block, n_block, k_block);

    if (!config.need_copy_a() && !config.need_copy_b())
        return res;

    ExpressionPtr copy_b_expr;
    const auto b_input_expr = brgemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<BrgemmCopyB>(b_input_expr->get_node())) {
        copy_b_expr = b_input_expr;
    } else if (ov::is_type<snippets::op::Buffer>(b_input_expr->get_node())) {
        const auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<BrgemmCopyB>(b_input_expr->get_node()))
            copy_b_expr = input_buffer_expr;
    }
    if (copy_b_expr) {
        copy_b_expr->get_input_port_descriptor(0)->set_subtensor({get_full_dim_value(), get_full_dim_value()});
        copy_b_expr->get_output_port_descriptor(0)->set_subtensor({get_full_dim_value(), get_full_dim_value()});
        if (config.need_compensations()) {
            const ov::snippets::VectorDims compensations_subtensor{1, n_block};
            OPENVINO_ASSERT(brgemm_expr->get_input_count() == 3, "Brgemm must have 3 inputs in case of compensations.");
            brgemm_expr->get_input_port_descriptor(2)->set_subtensor(compensations_subtensor);
            copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
        }
    }

    ExpressionPtr copy_a_expr;
    if (config.need_copy_a()) {
        const auto a_input_expr = brgemm_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<BrgemmCopyA>(a_input_expr->get_node())) {
            copy_a_expr = a_input_expr;
        } else if (ov::is_type<snippets::op::Buffer>(a_input_expr->get_node())) {
            const auto input_buffer_expr = a_input_expr->get_input_port_connector(0)->get_source().get_expr();
            if (ov::is_type<BrgemmCopyA>(a_input_expr->get_node()))
                copy_a_expr = input_buffer_expr;
        }
        OPENVINO_ASSERT(copy_a_expr);
    }

    if (config.need_wsp()) {
        move_new_memory_buffer(linear_ir, brgemm_it);
        auto buffer_it = std::prev(brgemm_it);
        buffer_it->get()->set_loop_ids(brgemm_expr->get_loop_ids());
    }

    if (copy_a_expr) {
        copy_a_expr->get_input_port_descriptor(0)->set_subtensor({m_block, ov::snippets::utils::get_full_dim_value()});
        copy_a_expr->get_output_port_descriptor(0)->set_subtensor({m_block, ov::snippets::utils::get_full_dim_value()});
        auto copy_a_expr_it = linear_ir.find(copy_a_expr);
        if (*copy_a_expr_it != *std::prev(brgemm_it)) {
            linear_ir.move(copy_a_expr_it, brgemm_it);
        }
        if (!is_full_dim_value(m_block)) {
            auto m_loop_id = brgemm_expr->get_loop_ids().front();
            copy_a_expr->set_loop_ids({m_loop_id});
        }
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    if (!is_full_dim_value(m_block)) {
        auto m_loop_id = brgemm_expr->get_loop_ids().front();
        if (config.need_compensations()) {
            const LoopPort default_port(brgemm_expr->get_input_port(1), false, 1);
            const std::vector<LoopPort> replacement_ports {default_port, LoopPort(brgemm_expr->get_input_port(2), false, 1)};
            loop_manager->get_loop_info<UnifiedLoopInfo>(m_loop_id)->replace_with_new_ports(default_port, replacement_ports);
        }
        if (config.need_copy_a()) {
            const LoopPort default_port(brgemm_expr->get_input_port(0), true, 1);
            const std::vector<LoopPort> replacement_ports {LoopPort(copy_a_expr->get_input_port(0), true, 1)};
            loop_manager->get_loop_info<UnifiedLoopInfo>(m_loop_id)->replace_with_new_ports(default_port, replacement_ports);
        }
    }

    return true;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov