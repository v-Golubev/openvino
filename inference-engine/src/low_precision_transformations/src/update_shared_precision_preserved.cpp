// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/update_shared_precision_preserved.hpp"

#include <assert.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

//std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> getParentInputRestrictions(
//    const std::shared_ptr<ngraph::Node> node) {
//    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> parentAttributes;
//    for (size_t index = 0ul; index < node->get_input_size(); index++) {
//        const Input<Node>& input = node->input(index);
//        auto inputNode = input.get_source_output().get_node()->shared_from_this();
//
//        const auto dequantization = NetworkHelper::getDequantization(node, index);
//        if (!dequantization.empty() &&
//            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
//            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
//            inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
//        }
//
//        if (NetworkHelper::isPrecisionPreserved(inputNode)) {
//            //for (const Input<Node>& input : inputNode->inputs()) {
//            //    auto& inputRtInfo = input.get_rt_info();
//            //    auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            //    if (inputAttributeIt != inputRtInfo.end()) {
//            //        const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(
//            //            inputAttributeIt->second);
//            //        parentAttributes.push_back(attribute);
//            //    }
//            //}
//
//            auto& inputRtInfo = inputNode->get_rt_info();
//            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            if (inputAttributeIt != inputRtInfo.end()) {
//                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
//                parentAttributes.push_back(attribute);
//            }
//        } else if (is_type<opset1::FakeQuantize>(inputNode)) {
//            const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
//            auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            if (attributeIt != outputPortRtInfo.end()) {
//                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attributeIt->second);
//                parentAttributes.push_back(attribute);
//            }
//        }
//    }
//    return parentAttributes;
//}
//
////void replaceAttributeInInputs(
////    std::shared_ptr<ngraph::Function> f,
////    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> newAttribute,
////    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> oldAttribute,
////    const std::shared_ptr<ngraph::Node>& initialNode) {
////    const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
////
////    std::set<std::shared_ptr<Node>> visited;
////    std::deque<std::shared_ptr<Node>> nodes;
////    nodes.emplace_back(initialNode);
////
////    //bool initialNodeIsNotInitialized = true;
////
////    while (!nodes.empty()) {
////        auto node = nodes.front();
////        nodes.pop_front();
////
////        if (visited.count(node) || is_type<op::Constant>(node)) {
////            continue;
////        }
////
////        visited.insert(node);
////
////        bool handleConnectedNodes = false;
////        if (is_type<opset1::FakeQuantize>(node)) {
////            for (auto& output : node->outputs()) {
////                auto& rt = output.get_rt_info();
////                if (node == initialNode) {
////                    rt[name] = newAttribute;
////                    handleConnectedNodes = true;
////                } else {
////                    auto it = rt.find(name);
////                    if (it != rt.end()) {
////                        const auto currentAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
////                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw1 = oldAttribute.get();
////                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw2 = currentAttribute.get();
////                        if (raw1 == raw2) {
////                            rt[name] = newAttribute;
////                        }
////                        handleConnectedNodes = true;
////                    }
////                }
////            }
////        } else {
////            for (size_t index = 0ul; index < node->get_input_size(); ++index) {
////                //auto getInput = [](const std::shared_ptr<Node>& node, const size_t index) -> const Input<Node> {
////                //    // TODO: isPrecisionPreserved
////                //    const auto dequantization = NetworkHelper::getDequantization(node, index);
////                //    if (!dequantization.empty() &&
////                //        (is_type<opset1::Convert>(dequantization.data.get_node())) &&
////                //        is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
////
////                //        const auto& targetInputs = dequantization.data.get_target_inputs();
////                //        if (targetInputs.size() == 1ul) {
////                //            return *targetInputs.begin();
////                //        }
////                //    }
////
////                //    return node->input(index);
////                //};
////
////                //auto input = getInput(node, index);
////
////                auto input = node->input(index);
////                auto& rt = input.get_rt_info();
////
////                if (node == initialNode) {
////                    rt[name] = newAttribute;
////                    handleConnectedNodes = true;
////                } else {
////                    auto it = rt.find(name);
////                    if (it != rt.end()) {
////                        const auto currentAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
////                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw1 = oldAttribute.get();
////                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw2 = currentAttribute.get();
////                        if (raw1 == raw2) {
////                            rt[name] = newAttribute;
////                        }
////                        handleConnectedNodes = true;
////                    }
////                }
////            }
////        }
////
////        if (!handleConnectedNodes) {
////            continue;
////        }
////
////        if (!is_type<opset1::FakeQuantize>(node)) {
////            for (size_t index = 0ul; index < node->get_input_size(); ++index) {
////                auto getInput = [](const std::shared_ptr<ngraph::Node>& node, const size_t index) {
////                    const auto dequantization = NetworkHelper::getDequantization(node, index);
////                    if (!dequantization.empty() &&
////                        (is_type<opset1::Convert>(dequantization.data.get_node())) &&
////                        is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
////                        const auto input = dequantization.data.get_node()->input(0);
////                        return input;
////                    }
////                    return node->input(index);
////                };
////
////                auto input = getInput(node, index);
////                const auto& input_node = input.get_source_output().get_node_shared_ptr();
////                if (visited.count(input_node) || is_type<op::Constant>(input_node)) {
////                    continue;
////                }
////
////                nodes.push_front(input_node);
////            }
////        }
////
////        for (auto& output : node->outputs()) {
////            for (auto& input_value : output.get_target_inputs()) {
////                const auto& output_node = input_value.get_node()->shared_from_this();
////                if (visited.count(output_node) || is_type<op::Constant>(output_node)) {
////                    continue;
////                }
////
////                nodes.push_front(output_node);
////            }
////        }
////    }
////}
//
//void handle(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::Node>& node) {
//    // TODO: possible need to add validation here to avoid not neccaassary actions for not preserved operations without precision limitations
//    const bool precisionPreserved = NetworkHelper::isPrecisionPreserved(node);
//
//    if (precisionPreserved) {
//        const auto parentRestrictions = getParentInputRestrictions(node);
//        if (parentRestrictions.empty()) {
//            return;
//        }
//
//        // TODO: there is limitation here: one operation - one output precision
//        // 1. merge parent inputs to one current output
//        auto resultAttribute = parentRestrictions[0];
//
//        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> toMerge = parentRestrictions;
//        toMerge.erase(toMerge.begin());
//        resultAttribute->merge(toMerge);
//
//        for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
//            const auto oldAttribute = parentRestrictions[index]->get();
//            //replaceAttributeInInputs(f, resultAttribute, parentRestrictions[index], node);
//
//            NetworkHelper::reassign<PrecisionsSharedValue, PrecisionsAttribute>(
//                resultAttribute->get()->sharedValue,
//                parentRestrictions[index]->get()->sharedValue->attributes);
//        }
//
//        auto& rt = node->get_rt_info();
//        rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//
//        //// 2. propagate
//        //if (is_type<opset1::FakeQuantize>(node)) {
//        //    auto& outputPortRtInfo = node->outputs()[0].get_rt_info();
//        //    outputPortRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//        //} else {
//        //    for (auto& input : node->inputs()) {
//        //        auto& rt = input.get_rt_info();
//        //        rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//        //    }
//        //}
//    }
//}

//bool ngraph::pass::low_precision::PropagateThroughPrecisionPreserved::run_on_function(std::shared_ptr<ngraph::Function> f) {
//    std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
//    for (auto it = nodes.begin(); it != nodes.end(); it++) {
//        const std::shared_ptr<Node> node = *it;
//        if (is_type<opset1::FakeQuantize>(node)) {
//            assert(node->get_output_size() == 1ul);
//            auto& outputRtInfo = node->output(0).get_rt_info();
//
//            auto attribute = make_shared_attribute<PrecisionsAttribute>(std::set<element::Type>{element::u8, element::i8});
//            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);
//            outputRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = attributeWrapper;
//            continue;
//        }
//
//        if (!NetworkHelper::isPrecisionPreserved(node)) {
//            for (auto& input : node->inputs()) {
//                auto parentNode = input.get_source_output().get_node_shared_ptr();
//
//                // TODO: move to method
//                auto getAttributes = [](const Input<Node>& nodeInput) {
//                    const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
//
//                    auto node = nodeInput.get_source_output().get_node_shared_ptr();
//                    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes;
//                    if (is_type<opset1::FakeQuantize>(node)) {
//                        // output
//                        auto& rt = nodeInput.get_source_output().get_rt_info();
//                        auto it = rt.find(name);
//                        if (it != rt.end()) {
//                            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
//                            attributes.push_back(attribute);
//                        }
//                    } else if (NetworkHelper::isPrecisionPreserved(node)) {
//                        // inputs
//                        for (auto input : node->inputs()) {
//                            auto& rt = input.get_rt_info();
//                            auto it = rt.find(name);
//                            if (it == rt.end()) {
//                                continue;
//                            }
//                            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
//                            attributes.push_back(attribute);
//                        }
//                    }
//
//                    return attributes;
//                };
//
//                auto& nodeRt = input.get_rt_info();
//
//                const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
//                const auto it = nodeRt.find(name);
//                if (it == nodeRt.end()) {
//                    continue;
//                }
//
//                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
//                std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes{ attribute};
//
//                auto parentAttributes = getAttributes(input);
//                if (parentAttributes.empty()) {
//                    continue;
//                }
//
//                for (auto& parentAttribute : parentAttributes) {
//                    parentAttribute->merge(attributes);
//                }
//
//                nodeRt[name] = parentAttributes[0];
//            }
//            continue;
//        }
//
//        handle(f, node);
//    }
//    return true;
//}
