// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/split_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool updatePrecisions,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        updatePrecisions ? values.lowPrecision : originalFunctionPrecision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        auto constant = std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision,
            values.subtractShape, values.subtractValues);
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(parent, constant);
        parent = subtract;
    }

    if (!values.multiplyValues.empty()) {
        auto constant = std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision,
            values.multiplyShape, values.multiplyValues);
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(parent, constant);
        parent = multiply;
    }
    auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, values.splitedAxis);
    const std::shared_ptr<ngraph::Node> split = std::make_shared<ngraph::opset1::Split>(parent, constant, values.numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < values.numSplit; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->get_output_as_single_output_node(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitTransformation");
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
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool updatePrecisions,
    const ExpectedValues& values) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(
        updatePrecisions ? values.lowPrecision : originalFunctionPrecision,
        ngraph::Shape(inputShape));

    auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, values.splitedAxis);
    const std::shared_ptr<ngraph::Node> split = std::make_shared<ngraph::opset1::Split>(input, constant, values.numSplit);

    std::vector<std::shared_ptr<ngraph::Node>> parents(values.numSplit);
    for (size_t i = 0; i < values.numSplit; ++i) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(split, originalFunctionPrecision);
        parents[i] = convert;
    }

    if (!values.subtractValues.empty()) {
        for (size_t i = 0; i < values.numSplit; ++i) {
            auto subConst = std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.subtractShape, values.subtractValues[i]);
            const std::shared_ptr<ngraph::Node> subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(parents[i], subConst);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(subtract, originalFunctionPrecision);
            parents[i] = subtract;
        }
    }

    for (size_t i = 0; i < values.numSplit; ++i) {
        auto mulConst = std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.multiplyShape, values.multiplyValues[i]);
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(parents[i], mulConst);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(multiply, originalFunctionPrecision);
        parents[i] = multiply;
    }

    ngraph::ResultVector results;
    for (size_t i = 0; i < values.numSplit; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(parents[i]));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
