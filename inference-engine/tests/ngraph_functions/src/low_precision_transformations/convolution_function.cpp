// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginal(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precisionBeforeDequantization,
    ngraph::builder::subgraph::DequantizationOperations dequantization,
    ngraph::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const auto weights = ngraph::opset1::Constant::create(
        weightsPrecision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        dequantizationOp,
        fakeQuantizeOnWeights.empty() ? weights->output(0) :
        ngraph::builder::makeFakeQuantize(
            weights, weightsPrecision,
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReference(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precisionBeforeDequantization,
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
    ngraph::element::Type precisionAfterOperation,
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
    ngraph::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights) {
    std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionAfterOperation,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    std::shared_ptr<ngraph::opset1::Convert> convert;
    if (!dequantizationBefore.convert.empty()) {
        convert = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convert>>(parent, precisionAfterOperation);
        parent = convert;
    }

    std::shared_ptr<ngraph::opset1::Subtract> subtract;
    std::shared_ptr<ngraph::opset1::Multiply> multiply;
    if (dequantizationAfter.empty()) {
        const std::shared_ptr<Node> quantizationBefore = makeDequantization(input, dequantizationBefore);
        parent = quantizationBefore;
    }
    else {
        if (!dequantizationBefore.subtract.empty()) {
            subtract = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
                parent,
                std::make_shared<ngraph::opset1::Constant>(
                    precisionAfterOperation,
                    // CPU workaround
                    Shape({ 1, inputShape[1], 1, 1 }),
                    dequantizationBefore.subtract.values.size() == 1ul ?
                    std::vector<float>(inputShape[1], dequantizationBefore.subtract.values[0]) :
                    dequantizationBefore.subtract.values));
            subtract->set_output_type(0, precisionAfterOperation, subtract->get_output_partial_shape(0));
            parent = subtract;
        }

        if (!dequantizationBefore.multiply.empty()) {
            multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Multiply>>(
                parent,
                std::make_shared<ngraph::opset1::Constant>(
                    precisionAfterOperation,
                    // CPU workaround
                    Shape({ 1, inputShape[1], 1, 1 }),
                    dequantizationBefore.multiply.values.size() == 1ul ?
                    std::vector<float>(inputShape[1], dequantizationBefore.multiply.values[0]) :
                    dequantizationBefore.multiply.values));
            multiply->set_output_type(0, precisionAfterOperation, multiply->get_output_partial_shape(0));
            parent = multiply;
        }
    }

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const std::shared_ptr<ngraph::opset1::Constant> weights = ngraph::opset1::Constant::create(
        precisionAfterOperation,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        parent,
        fakeQuantizeOnWeights.empty() ?
        weights->output(0) :
        ngraph::builder::makeFakeQuantize(
            weights->output(0),
            precisionAfterOperation,
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(convolution, dequantizationAfter);

    input = as_type_ptr<ngraph::opset1::Parameter>(replace_node(
        input,
        std::make_shared<ngraph::opset1::Parameter>(
            precisionBeforeDequantization,
            ngraph::Shape(inputShape))));

    if (subtract != nullptr) {
        if (dequantizationAfter.empty()) {
            replace_node(
                subtract->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(subtract->get_input_node_shared_ptr(1), precisionAfterOperation));
        } else {
            replace_node(
                subtract->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(subtract->get_input_node_shared_ptr(1), precisionBeforeDequantization));
        }
    }

    if (multiply != nullptr) {
        if (dequantizationAfter.empty()) {
            replace_node(
                multiply->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(multiply->get_input_node_shared_ptr(1), precisionAfterOperation));
        } else {
            replace_node(
                multiply->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(multiply->get_input_node_shared_ptr(1), precisionBeforeDequantization));
        }
    }

    replace_node(
        weights,
        ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(weights, weightsPrecision));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
