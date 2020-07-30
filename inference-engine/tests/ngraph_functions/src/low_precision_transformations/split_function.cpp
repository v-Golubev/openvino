// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/split_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const size_t numSplits) {
        const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
            precisionBeforeDequantization,
            ngraph::Shape(inputShape));

        const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
        const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
        const std::shared_ptr<Node> split = std::make_shared<ngraph::opset1::Split>(dequantizationOp, constant, numSplits);

        ngraph::ResultVector results;
        for (size_t i = 0; i < numSplits; ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(split->get_output_as_single_output_node(i)));
        }
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
    }

std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    int64_t splitedAxis, size_t numSplit) {
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

    auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const std::shared_ptr<ngraph::opset1::Split> split = std::make_shared<ngraph::opset1::Split>(fq, constant, numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->get_output_as_single_output_node(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ngraph::Function> SplitFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const size_t numSplit) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        ngraph::Shape(inputShape));

    std::shared_ptr<ngraph::opset1::Split> split;
    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    split = std::make_shared<ngraph::opset1::Split>(input, constant, numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        const std::shared_ptr<Node> quantizationOpAfter =
            makeDequantization(split->get_output_as_single_output_node(i), dequantizationAfter[i]);
        results.push_back(std::make_shared<ngraph::opset1::Result>(quantizationOpAfter));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
