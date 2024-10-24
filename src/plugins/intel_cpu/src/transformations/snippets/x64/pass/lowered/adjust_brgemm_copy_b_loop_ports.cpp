// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "adjust_brgemm_copy_b_loop_ports.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {

bool pass::AdjustBrgemmCopyBLoopPorts::update_loop_info(const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& loop_info) {
    OPENVINO_ASSERT(loop_info, "Invalid loop info pointer");
    snippets::lowered::UnifiedLoopInfo::LoopPortDesc *copy_b_loop_desc = nullptr;
    bool first_port = true;
    bool first_port_incremented = false;
    ov::element::Type precision;
    auto caller = [&](snippets::lowered::LoopPort &loop_port,
                      snippets::lowered::UnifiedLoopInfo::LoopPortDesc &loop_desc) {
        const auto& p = *loop_port.expr_port;
        if (p.get_type() == snippets::lowered::ExpressionPort::Input &&
            p.get_index() == 1) {
            const auto& node = p.get_expr()->get_node();
            if (auto brg = as_type_ptr<BrgemmCPU>(node)) {
                if (brg->get_config().need_copy_b()) {
                    copy_b_loop_desc = &loop_desc;
                    precision = node->get_input_element_type(1);
                }
            }
        }
        if (first_port) {
            first_port = false;
            first_port_incremented = loop_port.is_incremented;
        }
    };
    loop_info->iterate_through_infos(caller);

    if (copy_b_loop_desc && precision != element::f32) {
        // K blocking loop
        if (first_port_incremented) {
            const auto ptr_incr = copy_b_loop_desc->ptr_increment;
            const auto blocked_shape_ptr_inc = brgemm_utils::repacking::compute_out_leading_dim(ptr_incr, precision);
            if (ptr_incr != 0 && ptr_incr != blocked_shape_ptr_inc) {
                copy_b_loop_desc->ptr_increment = blocked_shape_ptr_inc;
                OPENVINO_ASSERT(copy_b_loop_desc->finalization_offset % ptr_incr == 0,
                                "Can't rescale finalization offsets");
                copy_b_loop_desc->finalization_offset = copy_b_loop_desc->ptr_increment *
                                                        (copy_b_loop_desc->finalization_offset / ptr_incr);
            }
        // N blocking loop
        } else {
            int64_t k_blk_size = 4 / static_cast<int64_t>(precision.size());
            copy_b_loop_desc->ptr_increment =
                    snippets::utils::dynamic_safe_mul(copy_b_loop_desc->ptr_increment, k_blk_size);
            copy_b_loop_desc->finalization_offset =
                    snippets::utils::dynamic_safe_mul(copy_b_loop_desc->finalization_offset, k_blk_size);
        }
        return true;
    }
    return false;
}

bool pass::AdjustBrgemmCopyBLoopPorts::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")

    bool modified = false;

    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        if (!is_type<BrgemmCopyB>(node))
            continue;
        const auto& loop_ids = expr->get_loop_ids();
        const auto& child_ports = expr->get_output_port(0).get_connected_ports();
        // Note: this pass should be executed before Loop insertion, so there is no LooEnd fake dependency
        OPENVINO_ASSERT(child_ports.size() == 1 &&
                        is_type<snippets::lowered::BufferExpression>(child_ports.begin()->get_expr()),
                        "BrgemmCopyB should have one BufferExpression child");
        auto grandchild_ports = child_ports.begin()->get_expr()->get_output_port(0).get_connected_ports();
        auto it = grandchild_ports.begin();
        while (it != grandchild_ports.end()) {
            const auto& port_node = it->get_expr()->get_node();
            if (is_type<intel_cpu::BrgemmCPU>(port_node)) {
                it++;
            } else {
                OPENVINO_ASSERT(is_type<snippets::op::LoopEnd>(port_node),
                                "Invalid grandchild of BrgemmCopyB");
                it = grandchild_ports.erase(it);
            }
        }
        for (const auto& target_port : grandchild_ports) {
            const auto &target_loop_ids = target_port.get_expr()->get_loop_ids();

            // If loop ids match, it means there is no blocking loop
            if (target_loop_ids == loop_ids)
                continue;
            OPENVINO_ASSERT(target_loop_ids.size() > loop_ids.size(), "Invalid BrgemmCopyB loop configuration");
            const auto &loop_map = linear_ir.get_loop_manager()->get_map();
            for (auto i = loop_ids.size(); i < target_loop_ids.size(); i++) {
                const auto &loop = loop_map.at(target_loop_ids[i]);
                auto uni_loop = ov::as_type_ptr<snippets::lowered::UnifiedLoopInfo>(loop);
                if (!uni_loop)
                    uni_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(loop)->get_unified_loop_info();
                if (!m_affected_loops.count(uni_loop) && update_loop_info(uni_loop)) {
                    m_affected_loops.insert(uni_loop);
                    modified = true;
                }
            }
        }
    }

    return modified;
}
} // namespace intel_cpu
} // namespace ov