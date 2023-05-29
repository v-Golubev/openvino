// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/loop_init.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

namespace {
void filter_ports(LoweredExprIR& linear_ir,
                  std::vector<LoweredExprPort>& loop_entries, std::vector<LoweredExprPort>& loop_exits) {
    std::vector<LoweredExprPort> new_loop_entries;
    std::vector<LoweredExprPort> new_loop_exits;
    new_loop_entries.reserve(loop_entries.size());
    new_loop_exits.reserve(loop_exits.size());

    std::set<std::shared_ptr<ov::Node>> loop_parents;
    for (const auto& loop_entry_point : loop_entries) {
        const auto& expr = loop_entry_point.expr;
        const auto port = loop_entry_point.port;
        const auto node = expr->get_node();
        if (is_type<op::Load>(node) || is_type<op::BroadcastLoad>(node)) {
            const auto& parent_expr = linear_ir.get_expr_by_output(expr->get_inputs()[port]).expr;
            const auto& parent = parent_expr->get_node();
            // Todo: Sometimes several Load in one Loop read data from the same Node.
            if (loop_parents.find(parent) == loop_parents.end()) {
                    loop_parents.insert(parent);
                    new_loop_entries.push_back(loop_entry_point);
            }
        }
    }

    for (const auto& loop_exit_point : loop_exits) {
        const auto& expr = loop_exit_point.expr;
        const auto& node = expr->get_node();
        if (is_type<op::Store>(node)) {
            new_loop_exits.push_back(loop_exit_point);
        }
    }

    loop_entries = new_loop_entries;
    loop_exits = new_loop_exits;
}

int64_t get_dim_stride(const size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim)
            break;
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
}  // namespace

LoopInit::LoopInit() : LinearIRTransformation() {}

std::vector<int64_t> LoopInit::init_ptr_increments(const std::vector<LoweredExprPort>& loop_inputs,
                                                   const std::vector<LoweredExprPort>& loop_outputs,
                                                   size_t dim_idx) const {
    std::vector<int64_t> ptr_increments;
    // Note: All loop inputs must have the same layout by definition.
    // If this doesn't hold, then we're trying to inject loops in the wrong place.
    const std::vector<size_t> loop_layout{
            !loop_inputs.empty() ? loop_inputs.front().expr->get_inputs()[0]->get_layout() :
            !loop_outputs.empty() ? loop_outputs.front().expr->get_outputs()[0]->get_layout() :
            std::vector<size_t>{}};
    // Note: Need to find max relevant dim expr to account for broadcasting, collect relevant_dims as well
    // Note: At the moment all loop_inputs and loop_outputs - are Load/Store ops in this method.
    //       So for example, we can call loop_input[i]->get_outputs().front() because Load have one output
    size_t max_relevant_dim_size = 0;
    for (const auto& loop_input : loop_inputs) {
        const auto& expr = loop_input.expr;
        int fake_loop_correction = expr->get_loop_ids().back() == LoweredExpr::LOOP_FAKE_ID ? -1 : 0;
        const auto out_td = expr->get_outputs().front();
        const auto& layout = out_td->get_layout();
        const auto& tensor = out_td->get_tensor();
        const auto& dim = *(layout.rbegin() + dim_idx + fake_loop_correction);
        max_relevant_dim_size = std::max(tensor[dim], max_relevant_dim_size);
    }
    for (const auto& loop_output : loop_outputs) {
        const auto& expr = loop_output.expr;
        bool has_fake_loop = expr->get_loop_ids().back() == LoweredExpr::LOOP_FAKE_ID;
        const auto in_td = expr->get_inputs().front();
        const auto& layout = in_td->get_layout();
        const auto& tensor = in_td->get_tensor();
        const auto& dim = *(layout.rbegin() + dim_idx - has_fake_loop);
        max_relevant_dim_size = std::max(tensor[dim], max_relevant_dim_size);
    }
    for (const auto& loop_input : loop_inputs) {
        const auto& expr = loop_input.expr;
        bool has_fake_loop = expr->get_loop_ids().back() == LoweredExpr::LOOP_FAKE_ID;
        const auto out_td = expr->get_outputs().front();
        const auto& layout = out_td->get_layout();
        const auto& tensor = out_td->get_tensor();
        const auto& dim = *(layout.rbegin() + dim_idx - has_fake_loop);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(tensor[dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dim, has_fake_loop ? loop_layout : layout, tensor);
        ptr_increments.push_back(ptr_increment);
    }
    // Note: Le already accounted for loop_input vs inside loops layout mismatch. So we need non-dense output
    // ptr_increments only if loop_input_layout doesn't match loop_output_layout
    for (const auto& loop_output : loop_outputs) {
        const auto& expr = loop_output.expr;
        bool has_fake_loop = expr->get_loop_ids().back() == LoweredExpr::LOOP_FAKE_ID;;
        const auto in_td = expr->get_inputs().front();
        const auto& layout = in_td->get_layout();
        const auto& tensor = in_td->get_tensor();
        const auto& dim = *(layout.rbegin() + dim_idx - has_fake_loop);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(tensor[dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dim, has_fake_loop ? loop_layout : layout, tensor);
        ptr_increments.push_back(ptr_increment);
    }

    return ptr_increments;
}

std::vector<int64_t> LoopInit::init_finalization_offsets(const std::vector<int64_t>& ptr_increments, size_t work_amount) const {
    std::vector<int64_t> finalization_offsets;
    for (const auto& ptr_incr : ptr_increments) {
        int64_t offset = -1 * ptr_incr * work_amount;
        finalization_offsets.push_back(offset);
    }
    return finalization_offsets;
}

std::vector<int64_t> LoopInit::init_element_type_sizes(const std::vector<LoweredExprPort>& loop_inputs,
                                                       const std::vector<LoweredExprPort>& loop_outputs) {
    std::vector<int64_t> element_types;
    element_types.reserve(loop_inputs.size() + loop_outputs.size());
    for (const auto& in : loop_inputs) {
        element_types.push_back(in.expr->get_node()->get_input_element_type(in.port).size());
    }
    for (const auto& out : loop_outputs) {
        element_types.push_back(out.expr->get_node()->get_output_element_type(out.port).size());
    }
    return element_types;
}

bool LoopInit::insertion(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                         size_t loop_id, size_t dim_idx, bool has_outer_loop) {
    auto loop_entries = loop_info->entry_exprs;
    auto loop_exits = loop_info->exit_exprs;
    const auto work_amount = loop_info->work_amount;
    const auto work_amount_increment = loop_info->increment;

    LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
    LoweredExprIR::LoweredLoopManager::get_loop_bounds(linear_ir, loop_entries, loop_exits, loop_begin_pos, loop_end_pos, loop_id);

    filter_ports(linear_ir, loop_entries, loop_exits);
    const auto ptr_increments = init_ptr_increments(loop_entries, loop_exits, dim_idx);
    const auto finalization_offsets = init_finalization_offsets(ptr_increments, work_amount);
    const auto io_data_sizes = init_element_type_sizes(loop_entries, loop_exits);

    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = std::make_shared<LoweredExpr>(loop_begin, std::vector<TensorDescriptorPtr>{});
    linear_ir.insert(loop_begin_pos, loop_begin_expr);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            loop_begin->output(0), work_amount, work_amount_increment, ptr_increments, finalization_offsets,
            io_data_sizes, loop_entries.size(), loop_exits.size());
    loop_end->has_outer_loop = has_outer_loop;

    std::vector<TensorDescriptorPtr> loop_end_inputs;
    for (const auto& expr_port : loop_entries)
        loop_end_inputs.push_back(expr_port.expr->get_inputs()[expr_port.port]);
    for (const auto& expr_port : loop_exits)
        loop_end_inputs.push_back(expr_port.expr->get_outputs()[expr_port.port]);
    loop_end_inputs.push_back(linear_ir.get_expr_by_node(loop_begin)->get_outputs().front());

    const auto& loop_end_expr = std::make_shared<LoweredExpr>(loop_end, loop_end_inputs);
    linear_ir.insert(loop_end_pos, loop_end_expr);
    return true;
}

bool LoopInit::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::LoopInit")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) ||
            ov::is_type<op::Buffer>(node) ||     // Need to cover Buffer
            ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            if (loop_id >= LoweredExpr::LOOP_NULL_ID)
                continue;
            bool need_to_insert = inserted_loops.find(loop_id) == inserted_loops.end();
            if (need_to_insert) {
                const auto loop_info = loop_manager->get_loop_info(loop_id);
                const bool has_outer_loop = i > 0 && inserted_loops.find(expr_loops[i - 1]) != inserted_loops.end();
                const auto status = insertion(linear_ir, loop_info, loop_id,
                                              loop_depth - 1 - i, has_outer_loop);
                if (status)
                    inserted_loops.insert(loop_id);  // save Loop ID
                inserted_loops.insert(loop_id);
            }
        }
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
