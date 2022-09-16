// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "ngraph/op/util/elementwise_args.hpp"
#include "snippets/op/fused_mul_add.hpp"
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/add.hpp>

#include <ngraph/runtime/host_tensor.hpp>

using namespace ngraph;

snippets::op::FusedMulAdd::FusedMulAdd(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c) : Op({ a, b, c }) {
    constructor_validate_and_infer_types();
}

bool snippets::op::FusedMulAdd::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> snippets::op::FusedMulAdd::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FusedMulAdd);
    check_new_args_count(this, new_args);
    return std::make_shared<FusedMulAdd>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void snippets::op::FusedMulAdd::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NGRAPH_CHECK(input_size == 3, "FusedMulAdd must have 3 inputs");
    NGRAPH_CHECK(get_output_size() == 1, "FusedMulAdd must have only 1 output");

    const auto element_type = get_input_element_type(0);
    auto pshape = get_input_partial_shape(0);
    for (size_t i = 1; i < input_size; ++i) {
        NODE_VALIDATION_CHECK(this,
                              element_type == get_input_element_type(i),
                              "Argument element types are inconsistent.");
        NODE_VALIDATION_CHECK(this,
                              pshape == get_input_partial_shape(i),
                              "Argument shapes are inconsistent.");
    }

    set_output_type(0, element_type, pshape);
}

bool snippets::op::FusedMulAdd::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    INTERNAL_OP_SCOPE(FusedMulAdd);
    const HostTensorVector mul_res(1ul);
    const auto multiply = std::make_shared<ngraph::op::v1::Multiply>();
    if (!multiply->evaluate(mul_res, { input_values.at(0), input_values.at(1) })) {
        return false;
    }

    const auto add = std::make_shared<ngraph::op::v1::Add>();
    return add->evaluate(output_values, { mul_res.at(0), input_values.at(2) });
}
