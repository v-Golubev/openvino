// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/split.hpp"
#include "ngraph/node.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
SplitTransformation::SplitTransformation(const Params& params) : LayerTransformation(params) {}

void SplitTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Split>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void SplitTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!LayerTransformation::canBeTransformed(context, m.get_match_root())) {
        return;
    }

    const std::shared_ptr<Node> split = separateInStandaloneBranch(m.get_match_root());
    auto dequantization = NetworkHelper::getDequantization(split);

    std::vector<Output<Node>> inputs(split->get_input_size());
    for (size_t i = 0; i < split->get_input_size(); ++i) {
        inputs[i] = split->get_input_node_shared_ptr(i);
    }
    const size_t dequantizationIndex = NetworkHelper::getInputIndex(dequantization.multiply, split);
    inputs[dequantizationIndex] = dequantization.data;
    std::shared_ptr<ngraph::Node> newSplit = split->clone_with_new_inputs(inputs);
    newSplit->set_friendly_name(split->get_friendly_name());
    newSplit->get_output_inputs(0);

    const ngraph::Shape subConstShape = dequantization.subtract ?
        dequantization.subtract->get_input_node_shared_ptr(1)->get_shape() : Shape{};
    std::vector<float> subValues = dequantization.subtract ? as_type_ptr<opset1::Constant>(
        dequantization.subtract->get_input_node_shared_ptr(1))->cast_vector<float>() : std::vector<float>();

    const ngraph::Shape mulConstShape = dequantization.multiply->get_input_node_shared_ptr(1)->get_shape();
    std::vector<float> mulValues = as_type_ptr<opset1::Constant>(
        dequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<float>();

    int64_t SplitedAxis = as_type_ptr<opset1::Constant>(split->get_input_node_shared_ptr(1))->cast_vector<int64_t>()[0];
    size_t axis = SplitedAxis > 0 ? SplitedAxis : split->get_input_shape(0).size() + SplitedAxis;
    size_t outputSize = split->get_output_size();

    std::vector<std::shared_ptr<ngraph::Node>> lastNodes(outputSize);
    ngraph::OutputVector replacement;
    for (const auto& output : newSplit->get_outputs()) {
        const std::shared_ptr<ngraph::Node> convert =
            dequantization.convert->clone_with_new_inputs({ output.get_output() });

        std::shared_ptr<ngraph::Node> subtract;
        if ((!subConstShape.empty()) && (subConstShape[axis] != 1)) {
            Shape newSubConstShape(subConstShape);
            newSubConstShape[axis] /= outputSize;

            std::vector<float> newSubValues(
                subValues.begin() + output.get_index() * newSubConstShape[axis],
                subValues.begin() + (output.get_index() + 1) * newSubConstShape[axis]);

            const auto subConst = std::make_shared<ngraph::opset1::Constant>(
                dequantization.subtract->get_input_element_type(1), newSubConstShape, newSubValues);
            subtract = dequantization.subtract ?
                std::make_shared<ngraph::opset1::Subtract>(convert, subConst) : nullptr;
        } else {
            subtract = dequantization.subtract ? std::make_shared<ngraph::opset1::Subtract>(convert,
                dequantization.subtract->get_input_node_shared_ptr(1)->clone_with_new_inputs({})) : nullptr;
        }

        std::shared_ptr<ngraph::Node> multiply;
        if ((!mulConstShape.empty()) && (mulConstShape[axis] != 1)) {
            Shape newMulConstShape(mulConstShape);
            newMulConstShape[axis] /= outputSize;

            std::vector<float> newMulValues(
                mulValues.begin() + output.get_index() * newMulConstShape[axis],
                mulValues.begin() + (output.get_index() + 1) * newMulConstShape[axis]);

            auto mul_const = std::make_shared<ngraph::opset1::Constant>(
                dequantization.multiply->get_input_element_type(1), newMulConstShape, newMulValues);
            multiply = std::make_shared<ngraph::opset1::Multiply>(
                dequantization.subtract ? subtract : convert, mul_const);
        } else {
            multiply = std::make_shared<ngraph::opset1::Multiply>(
                dequantization.subtract ? subtract : convert,
                dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}));
        }

        lastNodes.push_back(multiply);
        replacement.push_back(multiply);
    }

    replace_node(split, replacement);
    updateOutputs(context, lastNodes, newSplit);
}

void SplitTransformation::updateOutputs(
    TransformationContext& context,
    std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
    std::shared_ptr<ngraph::Node> originalNode) const {
    const size_t outputSize = context.network->get_output_size();
    if (outputSize == 1) {
        updateOutput(context, lastNodes[0], originalNode);
    } else {
        const std::string originalName = originalNode->get_friendly_name();
        for (auto& lastNode : lastNodes) {
            for (size_t i = 0; i < outputSize; ++i) {
                std::shared_ptr<ngraph::Node> result = context.network->get_output_op(i);
                std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
                if (outputNode.get() == lastNode.get()) {
                    std::ostringstream oss;
                    oss << i;
                    originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
                    lastNode->set_friendly_name(originalName + "." + oss.str());
                    break;
                }
            }
        }
    }
}

bool SplitTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
