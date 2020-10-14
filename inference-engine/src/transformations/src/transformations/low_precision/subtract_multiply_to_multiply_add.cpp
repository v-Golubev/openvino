﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/subtract_multiply_to_multiply_add.hpp"

#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void SubtractMultiplyToMultiplyAddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

FakeQuantizeDequantization get(const std::shared_ptr<Node> node) {
    Output<Node> dataNode = node;

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = is_type<opset1::Constant>(
        dataNode.get_node_shared_ptr()->get_input_node_shared_ptr(1)) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode.get_node_shared_ptr()) :
        nullptr;
    if (multiply != nullptr) {
        dataNode = multiply->get_input_source_output(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = (dataNode.get_node_shared_ptr()->get_input_size() > 1ul)
        && is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(1)) ?
            as_type_ptr<opset1::Subtract>(dataNode.get_node_shared_ptr()) :
            nullptr;
    if (subtract != nullptr) {
        dataNode = subtract->get_input_source_output(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode.get_node_shared_ptr());
    if (convert != nullptr) {
        dataNode = convert->get_input_source_output(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

bool SubtractMultiplyToMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    multiply = separateInStandaloneBranch(multiply);
    FakeQuantizeDequantization dequantization = get(multiply);

    const element::Type precisionBeforeDequantization = dequantization.convert == nullptr ?
        (dequantization.subtract == nullptr ?
            dequantization.multiply->get_input_element_type(0) :
            dequantization.subtract->get_input_element_type(0)) :
        dequantization.convert->get_input_element_type(0);

    const element::Type precisionAfterDequantization = dequantization.subtract == nullptr ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);

    if (dequantization.empty()) {
        return false;
    }

    auto lastNew = dequantization.data;
    element::Type lastNewPrecision = precisionBeforeDequantization;
    std::shared_ptr<Node> lastPrevious = dequantization.multiply != nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.multiply) :
        dequantization.subtract;

    {
        const std::shared_ptr<Node> multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
                std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
                ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(multiplyConstant, element::f32).get());

            auto lastNewPtr = lastNew.get_node_shared_ptr();
            NetworkHelper::setOutDataPrecision(as_type_ptr<opset1::Multiply>(lastNewPtr), precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<DequantizationMultiply>(lastNew, multiplyConstant);
        }
        if (dequantization.multiply != nullptr) {
            auto lastNewPtr = lastNew.get_node_shared_ptr();
            NetworkHelper::copyInfo(dequantization.multiply, lastNewPtr);
        }

        lastNewPrecision = precisionAfterDequantization;
    }

    if (dequantization.subtract != nullptr) {
        std::shared_ptr<Node> originalSubtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);

        std::shared_ptr<Node> subtractConstant = fold<opset1::Multiply>(
            fold<opset1::Multiply>(
                fold<opset1::Convert>(originalSubtractConstant, precisionAfterDequantization),
                std::make_shared<opset1::Constant>(precisionAfterDequantization, Shape{}, std::vector<float>{ -1.f })),
            fold<opset1::Convert>(dequantization.multiply->get_input_node_shared_ptr(1), precisionAfterDequantization));

        if (is_type<opset1::Constant>(subtractConstant)) {
            std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(subtractConstant);
            if (NetworkHelper::isScalarLike(constant)) {
                subtractConstant = NetworkHelper::toScalar(constant);
            }
        }

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<DequantizationAdd>>(
                std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
                ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(subtractConstant, element::f32).get());

            auto lastNewPtr = lastNew.get_node_shared_ptr();
            NetworkHelper::setOutDataPrecision(as_type_ptr<opset1::Add>(lastNewPtr), precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<DequantizationAdd>(lastNew, subtractConstant);
        }

        auto lastNewPtr = lastNew.get_node_shared_ptr();
        NetworkHelper::copyInfo(dequantization.subtract, lastNewPtr);

        lastNewPrecision = precisionAfterDequantization;
    }

    const std::shared_ptr<Node> lastOriginal = dequantization.multiply == nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.subtract) :
        dequantization.multiply;
    const std::shared_ptr<Node> lastNewPtr = lastNew.get_node_shared_ptr();
    replace_node(lastOriginal, lastNewPtr);

    updateOutput(context, lastNewPtr, lastPrevious);
    return true;
}

bool SubtractMultiplyToMultiplyAddTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    FakeQuantizeDequantization dequantization = get(op);
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return false;
    }

    if (((dequantization.subtract == nullptr) || (!dequantization.subtract->get_rt_info().count("DEQUANTIZATION"))) &&
        (!dequantization.multiply->get_rt_info().count("DEQUANTIZATION"))) {
        return false;
    }

    return
        ((dequantization.subtract == nullptr) || FakeQuantizeDequantization::checkElementwise(dequantization.subtract)) &&
        FakeQuantizeDequantization::checkElementwise(dequantization.multiply);
}

bool SubtractMultiplyToMultiplyAddTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
