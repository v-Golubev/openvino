// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/reshape.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Reshape::Reshape(const Output<Node>& arg, ov::PartialShape target_shape)
    : Op({arg}), m_target_shape(std::move(target_shape)) {
    constructor_validate_and_infer_types();
}

void Reshape::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_target_shape);
}

std::shared_ptr<Node> Reshape::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Reshape);
    check_new_args_count(this, new_args);
    return std::make_shared<Reshape>(new_args.at(0), m_target_shape);
}

bool Reshape::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("target_shape", m_target_shape);
    return true;
}

const ov::PartialShape& Reshape::get_target_shape() const {
    return m_target_shape;
}

void Reshape::set_target_shape(ov::PartialShape shape) {
    m_target_shape = std::move(shape);
}

ReshapeWithOrder::ReshapeWithOrder(const Output<Node>& arg, std::vector<size_t> order)
    : Op({arg}) {
    custom_constructor_validate_and_infer_types(std::move(order));
}

void ReshapeWithOrder::custom_constructor_validate_and_infer_types(std::vector<size_t> order) {
    INTERNAL_OP_SCOPE(ReshapeWithOrder_constructor_validate_and_infer_types);

    const auto& input_pshape = get_input_partial_shape(0);
    OPENVINO_ASSERT(input_pshape.rank().is_static() && input_pshape.size() == order.size(),
                   "Incompatible shape and order sizes");

    // During ctor call, ReshapeWithOrder doesn't know his port descriptors.
    // So we use explicit layouts from parameters
    set_output_type(0, get_input_element_type(0), ov::snippets::utils::get_planar_pshape(input_pshape, order));
}

void ReshapeWithOrder::validate_and_infer_types() {
    const auto& input_pshape = get_input_partial_shape(0);
    const auto order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    OPENVINO_ASSERT(input_pshape.rank().is_static() && input_pshape.size() == order.size(),
                    "Incompatible shape and order sizes");
    const auto output_pshape = utils::get_planar_pshape(get_input_partial_shape(0), order);
    set_output_type(0, get_input_element_type(0), output_pshape);
}

std::shared_ptr<Node> ReshapeWithOrder::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ReshapeWithOrder);
    check_new_args_count(this, new_args);
    const auto& order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    return std::make_shared<ReshapeWithOrder>(new_args.at(0), order);
}

bool ReshapeWithOrder::visit_attributes(AttributeVisitor& visitor) {
    auto order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    visitor.on_attribute("target_order", order);
    return true;
}

}// namespace op
}// namespace snippets
}// namespace ov