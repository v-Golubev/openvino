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

    auto iterate_through_ports = [&](const std::vector<size_t>& loop_ids,
                                     std::set<ov::snippets::lowered::ExpressionPort>& ports) {
        for (const auto& target_port : ports) {
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
    };

    for (const auto& expr : linear_ir) {
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(expr->get_node());
        if (!brgemm || !brgemm->get_config().need_copy_b())
            continue;
        const auto& input_connector = expr->get_input_port_connector(1);
        auto parent_out_ports = input_connector->get_consumers();
        const auto& parent_loop_ids = input_connector->get_source().get_expr()->get_loop_ids();
        iterate_through_ports(parent_loop_ids, parent_out_ports);
    }

    return modified;
}
} // namespace intel_cpu
} // namespace ov