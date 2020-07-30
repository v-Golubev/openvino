// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/variadic_split_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> VariadicSplitFunction::getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths) {
        const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
            precisionBeforeDequantization,
            ngraph::Shape(inputShape));

        const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
        const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
        const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
        const std::shared_ptr<Node> variadicSplit = std::make_shared<ngraph::opset1::VariadicSplit>(dequantizationOp, constantAxis, constantLengths);

        ngraph::ResultVector results;
        for (size_t i = 0; i < splitLengths.size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(variadicSplit->get_output_as_single_output_node(i)));
        }
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitFunction");
    }

std::shared_ptr<ngraph::Function> VariadicSplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fq = fakeQuantize.empty() ? nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            originalFunctionPrecision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
    const std::shared_ptr<Node> variadicSplit = std::make_shared<ngraph::opset1::VariadicSplit>(input, constantAxis, constantLengths);

    ngraph::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(variadicSplit->get_output_as_single_output_node(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitFunction");
}

std::shared_ptr<ngraph::Function> VariadicSplitFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionAfterOperation,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const std::vector<size_t>& splitLengths) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionAfterOperation,
        ngraph::Shape(inputShape));

    const auto constantAxis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const auto constantLengths = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ splitLengths.size() }, splitLengths);
    const std::shared_ptr<Node> variadicSplit = std::make_shared<ngraph::opset1::VariadicSplit>(input, constantAxis, constantLengths);

    ngraph::ResultVector results;
    for (size_t i = 0; i < splitLengths.size(); ++i) {
        const std::shared_ptr<Node> quantizationOpAfter =
            makeDequantization(variadicSplit->get_output_as_single_output_node(i), dequantizationAfter[i]);
        results.push_back(std::make_shared<ngraph::opset1::Result>(quantizationOpAfter));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "VariadicSplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
