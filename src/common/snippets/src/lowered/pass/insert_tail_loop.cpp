// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
LinearIR::container InsertTailLoop::copy_loop(const LinearIR& linear_ir, const size_t loop_id) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto original_loop_info = loop_manager->get_loop_info(loop_id);
    auto new_entry_points = original_loop_info->entry_points;
    auto new_exit_points = original_loop_info->exit_points;

    auto update_loop_ports = [](const ExpressionPtr& expr,
                                const ExpressionPtr& tail_expr,
                                std::vector<LinearIR::LoopManager::LoopPort>& ports) {
        auto find_if_predicate = [&](const LinearIR::LoopManager::LoopPort& port) {
            return port.expr_port->get_expr()->get_node() == expr->get_node();
        };
        auto pos = std::find_if(ports.begin(), ports.end(), find_if_predicate);
        while (pos != ports.end()) {
            pos->expr_port = std::make_shared<ExpressionPort>(tail_expr, pos->expr_port->get_type(), pos->expr_port->get_index());
            pos = std::find_if(pos, ports.end(), find_if_predicate);
        }
    };

    auto update_loop_info = [&](const ExpressionPtr& expr, const ExpressionPtr& new_expr) {
        const auto node = expr->get_node();
        // Loop begin/end ops can't be loop ports
        if (ov::is_type<op::LoopBase>(node))
            return;
        // Clone loop ports from original loop info to tail loop info
        update_loop_ports(expr, new_expr, new_entry_points);
        update_loop_ports(expr, new_expr, new_exit_points);

        // Update loop info of all inner loops with new loop ports
        const auto loop_ids = expr->get_loop_ids();
        auto cur_id_pos = std::find(loop_ids.begin(), loop_ids.end(), loop_id);
        std::vector<size_t> inner_loop_ids(loop_ids.begin(), cur_id_pos);
        for (size_t i = 0; i < expr->get_input_count(); ++i)
            loop_manager->update_loops_port(inner_loop_ids, expr->get_input_port(i), {expr->get_input_port(i), new_expr->get_input_port(i)}, true);
        for (size_t i = 0; i < expr->get_output_count(); ++i)
            loop_manager->update_loops_port(inner_loop_ids, expr->get_output_port(i), {expr->get_output_port(i), new_expr->get_output_port(i)}, false);
    };

    LinearIR::constExprIt loop_begin_pos, loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, loop_id, loop_begin_pos, loop_end_pos, true);
    const auto loop_copy_range = LinearIR::deep_copy_range(loop_begin_pos, std::next(loop_end_pos), update_loop_info);
    const auto new_loop_begin_pos = loop_copy_range.begin();
    const auto new_loop_end_pos = loop_copy_range.end();
    const auto new_id = loop_manager->mark_loop_with_old_loop_replacement(std::next(new_loop_begin_pos),
                                                                          std::prev(new_loop_end_pos),
                                                                          original_loop_info->work_amount,
                                                                          original_loop_info->increment,
                                                                          new_entry_points,
                                                                          new_exit_points,
                                                                          loop_id);
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_end_pos)->get()->get_node());
    loop_end->set_id(new_id);
    return loop_copy_range;
}

std::shared_ptr<op::LoopEnd> InsertTailLoop::create_tail_loop(LinearIR& linear_ir,
                                                              LinearIR::constExprIt vector_begin,
                                                              LinearIR::constExprIt vector_end,
                                                              LinearIR::constExprIt& tail_begin,
                                                              LinearIR::constExprIt& tail_end,
                                                              const std::shared_ptr<op::LoopEnd>& vector_loop_end,
                                                              bool need_vector_loop,
                                                              size_t tail_size,
                                                              const std::vector<int64_t>& tail_finalization_offsets) {
    // tail is required => transform the body into a tail representation
    // tail loop is fake loop because for tail we should calculate only
    // finalization offsets which are supported by LoopEnd.
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto original_loop_id = vector_loop_end->get_id();
    const auto& original_loop_info = loop_manager->get_loop_info(original_loop_id);
    auto tail_loop_info = original_loop_info;
    if (need_vector_loop) {
        const auto new_loop_range = copy_loop(linear_ir, original_loop_id);
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_range.end())->get()->get_node());
        loop_end->set_work_amount(tail_size);
        loop_end->set_increment(tail_size);
        tail_loop_info = loop_manager->get_loop_info(loop_end->get_id());
        tail_loop_info->work_amount = tail_size;
        tail_loop_info->increment = tail_size;

        tail_begin = linear_ir.insert(vector_end, new_loop_range.begin(), new_loop_range.end());
        tail_end = vector_end;
    } else {
        tail_begin = vector_begin;
        tail_end = vector_end;
    }

    auto update_subtensors = [&](const std::vector<LinearIR::LoopManager::LoopPort>& ports) {
        for (const auto& port : ports) {
            if (port.is_incremented) {
                auto desc = port.expr_port->get_descriptor_ptr();
                auto subtensor_shape = desc->get_subtensor();
                if (subtensor_shape.size() > port.dim_idx) {
                    *(subtensor_shape.rbegin() + port.dim_idx) = tail_size;
                    desc->set_subtensor(subtensor_shape);
                }
            }
        }
    };
    update_subtensors(tail_loop_info->entry_points);
    update_subtensors(tail_loop_info->exit_points);

    // We have to check the loop body for any nested loops that work on the same dimension
    // and rescale their work_amount and increment accordingly
    if (original_loop_info->outer_splited_loop) {
        const auto current_dim_idx = original_loop_info->get_dim_idx();
        OPENVINO_ASSERT(current_dim_idx != SIZE_MAX, "Outer splitted loop unexpectedly iterates by several dimension indices");
        for (auto it = std::next(tail_begin); it != std::prev(tail_end); ++it) {
            const auto& expr = *it;
            const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
            if (!inner_loop_end)
                continue;
            const auto inner_loop_info = loop_manager->get_loop_info(inner_loop_end->get_id());
            const auto inner_dim_idx = inner_loop_info->get_dim_idx();
            if (inner_dim_idx != current_dim_idx)
                continue;
            const auto inner_loop_begin = inner_loop_end->get_loop_begin();
            const auto inner_tail_work_amount = static_cast<int64_t>(inner_loop_end->get_work_amount());
            const auto inner_tail_increment = inner_loop_end->get_increment();
            auto inner_finalization_offsets = inner_loop_end->get_finalization_offsets();
            for (auto& offset : inner_finalization_offsets) {
                offset = offset / inner_tail_work_amount * static_cast<int64_t>(tail_size);
            }
            inner_loop_end->set_work_amount(tail_size);
            inner_loop_end->set_increment(std::min(inner_tail_increment, tail_size));
            inner_loop_end->set_finalization_offsets(inner_finalization_offsets);
            const auto inner_loop_begin_it = std::find(tail_begin, it, linear_ir.get_expr_by_node(inner_loop_begin));
            const auto inner_loop_end_it = std::next(tail_end);
            OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
            tail_transformations(linear_ir, inner_loop_begin_it, inner_loop_end_it, tail_size);
        }
    }

    tail_transformations(linear_ir, tail_begin, tail_end, tail_size);
    std::shared_ptr<op::LoopEnd> tail_loop_end = ov::as_type_ptr<op::LoopBegin>((*tail_begin)->get_node())->get_loop_end();
    tail_loop_end->set_increment(tail_size);
    // ptr increments were set to the old increment, need to update them in accordance with the new one
    tail_loop_end->set_work_amount(tail_size);
    tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
    tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;
    const auto new_vector_loop_wa = original_loop_info->work_amount - tail_size;
    original_loop_info->work_amount = new_vector_loop_wa;
    vector_loop_end->set_work_amount(new_vector_loop_wa);
    return tail_loop_end;
}

void InsertTailLoop::tail_transformations(LinearIR& linear_ir,
                                          LinearIR::constExprIt tail_begin,
                                          LinearIR::constExprIt tail_end,
                                          const size_t tail_size) {
    const auto& config = linear_ir.get_config();
    auto insertFill = [tail_size](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        std::shared_ptr<ov::Node> fill = nullptr;
        auto& rt = input.get_rt_info();
        auto fill_rt = rt.find("set_fill");
        if (fill_rt != rt.end()) {
            const auto fill_value = fill_rt->second.as<uint32_t>();
            fill = std::make_shared<ov::snippets::op::Fill>(input.get_source_output(), tail_size, fill_value);
            input.get_node()->set_argument(input.get_index(), fill);
        }
        return fill;
    };

    for (auto expr_it = std::next(tail_begin); expr_it != tail_end; expr_it++) {
        // Skip inner Loops
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(expr_it->get()->get_node());
        if (loop_begin) {
            expr_it = linear_ir.find(expr_it, tail_end, linear_ir.get_expr_by_node(loop_begin->get_loop_end()));
            continue;
        }
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (size_t i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    const auto& input = expr->get_input_port_connector(i);
                    const auto consumers = input->get_consumers();
                    auto fill_expr = linear_ir.create_expression(fill, {input});
                    linear_ir.insert(expr_it, fill_expr);
                    linear_ir.replace_input(consumers, fill_expr->get_output_port_connector(0));
                    // in_reg == out_reg since we want to modify vector reg inplace
                    const auto reg = expr->get_input_port_descriptor(0)->get_reg();
                    fill_expr->get_input_port_descriptor(0)->set_reg(reg);
                    fill_expr->get_output_port_descriptor(0)->set_reg(reg);
                    fill_expr->set_loop_ids(expr->get_loop_ids());
                }
            }
        } else if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::op::MemoryAccess>(op)) {
            for (const auto p : memory_access->get_memory_access_input_ports()) {
                const auto port = p.first;
                if (memory_access->get_input_count(port) > 1) {
                    memory_access->set_input_count(tail_size, port);
                }
            }
            for (const auto p : memory_access->get_memory_access_output_ports()) {
                const auto port = p.first;
                if (memory_access->get_output_count(port) > 1) {
                    memory_access->set_output_count(tail_size, port);
                }
            }
        }
    }
}

bool InsertTailLoop::optimize_single_evaluation(const std::shared_ptr<op::LoopEnd>& loop) {
    // *1* solo vector/tail loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/tail loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* vector loop(s) + one tail loop
    //      => vector as usual, tail depends on outer loop, see *1* and *2*
    if (loop->get_work_amount() >= 2 * loop->get_increment())
        return false;

    std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
    const auto& ptr_increments = loop->get_ptr_increments();
    const auto work_amount_incr = static_cast<int64_t>(loop->get_increment());
    for (size_t i = 0; i < new_finalization_offsets.size(); i++) {
        new_finalization_offsets[i] += ptr_increments[i] * work_amount_incr;
    }
    loop->set_finalization_offsets(new_finalization_offsets);
    loop->set_evaluate_once(true);
    return true;
}

bool InsertTailLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::insertTailLoop")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool modified = false;

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!loop_end)
            continue;

        const auto loop_info = loop_manager->get_loop_info(loop_end->get_id());
        if (loop_info->fst_iter_handler != nullptr) {
            modified |= loop_info->fst_iter_handler(linear_ir, expr_it);
        }

        if (loop_end->get_evaluate_once() == true)
            continue;

        const auto work_amount = loop_end->get_work_amount();
        const auto increment = loop_end->get_increment();
        const auto tail_size = work_amount % increment;
        const auto need_tail = tail_size != 0;
        const auto need_vector_loop = work_amount >= increment;
        // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
        // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + tail)
        const auto tail_finalization_offsets = need_tail ? loop_end->get_finalization_offsets() : std::vector<int64_t>{};
        // vector loops are required => Just copy the body, original loop is already a vector one
        if (need_vector_loop) {
            // Note that finalization offsets should be applied after the last iteration.
            // So if there is a tail, then we should apply offsets after it, but not now.
            if (need_tail)
                loop_end->set_finalization_offsets(std::vector<int64_t>(tail_finalization_offsets.size(), 0));

            optimize_single_evaluation(loop_end);
        }

        // tail is required => transform the body into a tail representation
        // tail loop is fake loop because for tail we should calculate only
        // finalization offsets which are supported by LoopEnd.
        if (need_tail) {
            const auto loop_begin = loop_end->get_loop_begin();
            const auto begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_begin));
            LinearIR::constExprIt tail_begin, tail_end;
            const auto tail_loop_end = create_tail_loop(linear_ir, begin_it, std::next(expr_it), tail_begin, tail_end,
                                                        loop_end, need_vector_loop, tail_size, tail_finalization_offsets);
            optimize_single_evaluation(tail_loop_end);
            // Skip new tail loop. Note: tail_end refs to the next expression after LoopEnd of tail
            expr_it = std::prev(tail_end);
        }
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

