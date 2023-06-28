// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/buffer_identification.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/rt_info.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
inline size_t index(size_t size, size_t row, size_t col) {
    return col + row * size;
}
} // namespace

std::vector<bool> BufferIdentification::create_adjacency_matrix(const ov::NodeVector& ops, const BufferIdentification::BufferSet& buffers) {
    // The sync point to check for adjency is Loop because only in Loop we increment pointers.
    // So if some Buffers in the one Loop have conflict (cannot be inplace: the same ptr increment and finalization offset)
    // they are called as adjacent
    const auto size = buffers.size();
    std::vector<bool> adj(size * size, false);
    for (size_t i = 0; i < size; ++i)
        adj[index(size, i, i)] = true;

    auto get_buffer_idx = [&](const std::shared_ptr<op::Buffer>& buffer) {
        const auto iter = std::find(buffers.cbegin(), buffers.cend(), buffer);
        NGRAPH_CHECK(iter != buffers.cend(), "Buffer wasn't find in Buffer system of Subgraph");
        return std::distance(buffers.cbegin(), iter);
    };

    auto update_adj_matrix = [&](const std::pair<std::shared_ptr<op::Buffer>, int64_t>& buffer,
                                 const std::pair<std::shared_ptr<op::Buffer>, int64_t>& neighbour_buffer) {
        if ((buffer.second != neighbour_buffer.second) ||
            (buffer.second > 0 && buffer.first->get_element_type().size() != neighbour_buffer.first->get_element_type().size())) {
            const auto buffer_idx = get_buffer_idx(buffer.first);
            const auto neighbour_idx = get_buffer_idx(neighbour_buffer.first);
            adj[index(size, neighbour_idx, buffer_idx)] = adj[index(size, buffer_idx, neighbour_idx)] = true;
        }
    };

    for (auto it = ops.begin(); it != ops.end(); ++it) {
        const auto& op = *it;
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(op);
        if (!loop_end)
            continue;

        const auto ptr_increments = loop_end->get_ptr_increments();

        const auto& loop_begin = loop_end->get_loop_begin();
        // Buffer -> ptr increment
        std::map<std::shared_ptr<op::Buffer>, int64_t> buffer_neighbours;

        for (size_t i = 0; i < loop_begin->input_values().size(); ++i) {
            const auto& input_value = loop_begin->input_value(i);
            auto loop_in = input_value.get_node_shared_ptr();
            auto port_idx = input_value.get_index();
            while (std::dynamic_pointer_cast<op::LoopBase>(loop_in)) {
                const auto source_output = loop_in->input_value(port_idx);
                loop_in = source_output.get_node_shared_ptr();
                port_idx = source_output.get_index();
            }

            if (const auto neighbour_buffer = ov::as_type_ptr<op::Buffer>(loop_in)) {
                buffer_neighbours[neighbour_buffer] = ptr_increments[i];
            }
        }

        for (const auto& output : loop_end->outputs()) {
            // check for first target input is enough for Buffer searching because operations can have only single Buffer per each output port as op
            const auto& target_inputs = output.get_target_inputs();
            auto consumer_in = *target_inputs.begin();
            auto port_idx = consumer_in.get_index();
            auto consumer = consumer_in.get_node()->shared_from_this();
            while (std::dynamic_pointer_cast<op::LoopBase>(consumer)) {
                const auto target_inputs = consumer->get_output_target_inputs(port_idx);
                auto consumer_in = *target_inputs.begin();
                port_idx = consumer_in.get_index();
                consumer = consumer_in.get_node()->shared_from_this();
            }

            if (const auto neighbour_buffer = ov::as_type_ptr<op::Buffer>(consumer)) {
                buffer_neighbours[neighbour_buffer] = ptr_increments[loop_begin->get_input_size() + output.get_index()];
            }
        }

        const auto& begin_it = std::find(ops.begin(), ops.end(), loop_begin);
        OPENVINO_ASSERT(begin_it != ops.end(), "LoopBegin hasn't been found!");

        for (auto loop_it = begin_it; loop_it != it; ++loop_it) {
            const auto& loop_op = *loop_it;
            if (const auto neighbour_buffer = ov::as_type_ptr<op::Buffer>(loop_op)) {
                buffer_neighbours[neighbour_buffer] = 0;
                continue;
            }

            // Skip inside Loops - these Loops have been already analysed
            if (const auto inner_loop_begin = ov::as_type_ptr<op::LoopBegin>(loop_op)) {
                const auto& inner_end_it = std::find(loop_it, ops.end(), inner_loop_begin);
                OPENVINO_ASSERT(inner_end_it != ops.end(), "LoopEnd hasn't been found!");
                loop_it = inner_end_it;
            }
        }

        for (auto buffer_it = buffer_neighbours.begin(); buffer_it != buffer_neighbours.end(); ++buffer_it) {
            for (auto neighbour_it = std::next(buffer_it); neighbour_it != buffer_neighbours.end(); ++neighbour_it) {
                update_adj_matrix(*buffer_it, *neighbour_it);
            }
        }
    }

    return adj;
}

std::map<size_t, BufferIdentification::BufferSet> BufferIdentification::coloring(BufferIdentification::BufferSet& buffers, std::vector<bool>& adj) {
    size_t color = 0;
    std::map<size_t, BufferIdentification::BufferSet> color_groups;
    const auto size = buffers.size();
    // If we have count of adjacent Buffers is equal to count of all Buffers,
    // it mean that Buffers aren't adjacent between them (they just have loops)
    if (static_cast<size_t>(std::count(adj.begin(), adj.end(), true)) == size) {
        color_groups[color] = buffers;
        return color_groups;
    }

    for (size_t i = 0; i < size; i++) {
        // The Buffer is already colored (visited) - skip
        if (!buffers[i])
            continue;

        const auto& buffer = buffers[i];
        color_groups[color].push_back(buffer); // Add to Color Group
        buffers[i] = nullptr;  // Remove from graph vertices

        // while Buffer i has not coloured non-neighbours
        // (row i contains 0)
        while (!std::accumulate(adj.begin() + i * size, adj.begin() + (i + 1) * size, true, std::logical_and<bool>())) {
            // Find first non-adjacent and non-visited (non-colored) Buffer to color him to the same color
            // NOTE: At the moment Snippets don't garantee that Buffer pointer won't be resetted after Loop execution.
            //       So we cannot reuse Buffer pointer at second time and don't allow the following case:
            //                   Buffer[0] -> ... -> Buffer[1] -> ... -> Buffer[0]
            //       To cover this case, we make force break when find first adjacent not-visitted `vertex`
            //       Notice, this case will be supported in new infrastructure with Linear IR
            size_t j = i + 1;
            bool force_break = false;
            for (; j < size; ++j) {
                if (adj[j + i * size] && buffers[j]) {
                    force_break = true;
                    break;
                }
                if (!adj[j + i * size] && buffers[j])
                    break;
            }

            // If we have to make force break or we don't have the corresponding non-adjacent and non-colored Buffers,
            // we should make break - all potential Buffers for the current color are already colored
            if (force_break || j == size)
                break;

            const auto& neighbour_buffer = buffers[j];
            color_groups[color].push_back(neighbour_buffer); // Add to Color Group
            buffers[j] = nullptr;  // Remove from graph vertices
            // Unite adjacency links:
            //    All the neighbors of Buffer `j` are added to the neighbors of Buffer `i` (the `vertices` are pulled together).
            //    The result is an updated i-th row of the adjacency matrix,
            //    in which 0 are only in columns with `vertex` numbers that are not adjacent to either the i-th or j-th `vertices`.
            //    Mathematically, this can be replaced by the operation of OR of Boolean vectors representing strings i and j.
            std::transform(adj.begin() + i * size, adj.begin() + (i + 1) * size, adj.begin() + j * size,
                           adj.begin() + i * size, std::logical_or<bool>());
        }

        color++;
    }

    return color_groups;
}

bool BufferIdentification::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(BufferIdentification);
    // Unite Buffers using Graph coloring algorithm.
    BufferIdentification::BufferSet buffers;

    const auto ops = model->get_ordered_ops();
    for (const auto& op : ops) {
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            buffers.push_back(buffer);
        }
    }

    // Creation of Adj matrix
    auto adj = create_adjacency_matrix(ops, buffers);

    // Graph coloring algorithm
    const auto color_groups = coloring(buffers, adj);

    for (const auto& pair : color_groups) {
        const auto color = pair.first;
        const auto& united_buffers = pair.second;
        for (const auto& buffer : united_buffers) {
            buffer->set_id(color);
        }
    }

    return true;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
