// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace intel_cpu {

/**
* @interface BrgemmCopyB
* @brief The operation for data repacking of Brgemm with input non-fp32 precisions.
         The CPU Generator uses oneDNN primitives for generation code of Brgemm.
         OneDNN requiers data repacking for second input of Brgemm with input non-fp32 precisions.
* @ingroup snippets
*/
class BrgemmCopyB : public snippets::modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("BrgemmCopyB", "SnippetsOpset");

    enum class Type {
        OnlyRepacking,     // Just data repacking - one output
        WithCompensations, // Repack data and caclulate compensations - 2 outputs (is needed for BrgemmCPU with compensations)
    };

    BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type = Type::OnlyRepacking,
                const size_t offset_in = 0lu, const size_t offset_out0 = 0lu, const size_t offset_out1 = 0lu,
                std::vector<size_t> layout_input = {});
    BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                const PortDescriptor& desc_in0, const PortDescriptor& desc_out0, const PortDescriptor& desc_out1,
                std::vector<size_t> layout_input = {});
    BrgemmCopyB() = default;

    size_t get_offset_in() const { return get_input_offset(0); }
    size_t get_offset_out() const { return get_output_offset(0); }
    size_t get_offset_compensations() const;

    Type get_type() const { return m_type; }
    element::Type get_src_element_type() const { return m_src_type; }
    bool is_with_compensations() const { return m_type == Type::WithCompensations; }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    bool has_evaluate() const override { return false; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public snippets::IShapeInferSnippets {
        std::vector<size_t> m_layout{};
        size_t m_num_outs = 1;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<snippets::VectorDimsRef>& input_shapes) override;
    };

private:
    void custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input = {});
    void validate_element_type(const ov::element::Type& element_type);

    Type m_type = Type::OnlyRepacking;
    element::Type m_src_type = ov::element::undefined;  // src element type of the corresponding BRGEMM
};
} // namespace intel_cpu

template <>
class AttributeAdapter<intel_cpu::BrgemmCopyB::Type> : public EnumAttributeAdapterBase<intel_cpu::BrgemmCopyB::Type> {
public:
    AttributeAdapter(intel_cpu::BrgemmCopyB::Type& value) : EnumAttributeAdapterBase<intel_cpu::BrgemmCopyB::Type>(value) {}
    OPENVINO_RTTI("AttributeAdapter<ov::intel_cpu::BrgemmCopyB::Type>");
};
} // namespace ov
