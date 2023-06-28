// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 *        If Buffer has a parent, the operation is for intermediate data storage - IntermediateMemory type.
 *        Otherwise, the operation is for allocation of new empty memory with shape `m_shape` - NewMemory type
 * @param m_type - type of Buffer: IntermediateMemory/NewMemory
 * @param m_shape - output allocation shape for Buffer with type NewMemory
 * @param m_id - Buffer ID in common Buffer system
 *        Notes:
 *               - All buffers in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @ingroup snippets
 */
class Buffer : public ngraph::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const ov::Shape& shape, size_t id = 0);
    Buffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape, size_t id = 0);
    Buffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank = -1, size_t id = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    enum Type {
        NewMemory,
        IntermediateMemory
    };

    void set_id(size_t id) { m_id = id; }

    size_t get_id() const { return m_id; }
    size_t get_byte_size() const;
    Type get_type() const { return m_type; }
    ov::Shape get_allocation_shape() const { return m_shape; }
    void set_allocation_shape(const ov::Shape& new_shape) { m_shape = new_shape; }

    bool is_intermediate_memory() const { return m_type == Type::IntermediateMemory; }
    bool is_new_memory() const { return m_type == Type::NewMemory; }

private:
    Type m_type = Type::IntermediateMemory;
    ov::Shape m_shape = {};
    size_t m_id = 0;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
