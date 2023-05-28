// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/mha.hpp"
#include "subgraph_mha.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace snippets {
namespace {
void input_generation(std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                      const std::vector<ov::Output<ov::Node>>& model_inputs,
                      const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    for (int i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), targetInputStaticShapes[i], 1.0f, 0.5f);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}
} // namespace

std::string MHA::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, withMul, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::partialShape2str({inputShapes[i]}) << "_";
    result << "Mul=" << withMul << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MHA::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    std::tie(inputShapes, withMul, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHAFunction(inputDynamicShapes, withMul);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHA::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    input_generation(inputs, function->inputs(), targetInputStaticShapes);
}

void MHASelect::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    std::tie(inputShapes, withMul, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHASelectFunction(inputDynamicShapes);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHASelect::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto model_inputs = function->inputs();
    for (auto& model_input : model_inputs) {
        const auto node_input = model_input.get_node_shared_ptr();
        const auto name = node_input->get_friendly_name();
        ov::Tensor tensor;
        int seed = 0;
        if (name.find("less") != std::string::npos) {
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape(), 5 + seed, -2, 10, seed);
            seed++;
        } else {
            tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), model_input.get_shape(), 1.0f, 0.5f);
        }
        inputs.insert({node_input, tensor});
    }
}

std::string MHAWOTranspose::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAWOTransposeParams> obj) {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul, transposedB_0;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, withMul, transposedB_0, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::partialShape2str({inputShapes[i]}) << "_";
    result << "Mul=" << withMul << "_";
    result << "TransposedB_0=" << withMul << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MHAWOTranspose::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    input_generation(inputs, function->inputs(), targetInputStaticShapes);
}

void MHAWOTranspose::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul, transposedB_0;
    std::tie(inputShapes, withMul, transposedB_0, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHAWOTransposeFunction(inputDynamicShapes, withMul, transposedB_0);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHAWOTransposeOnInputs::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul, transposedB_0;
    std::tie(inputShapes, withMul, transposedB_0, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHAWOTransposeOnInputsFunction(inputDynamicShapes, withMul, transposedB_0);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}


TEST_P(MHA, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(MHASelect, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTranspose, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTransposeOnInputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}


} // namespace snippets
} // namespace test
} // namespace ov
