// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/matmul_perf.hpp"
#include "subgraph_matmul.hpp"
#include "subgraph_mha.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string MatMulPerf::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulPerfParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    bool snippets_enabled;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_types, num_nodes, num_subgraphs, snippets_enabled, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i <<"]=" << CommonTestUtils::partialShape2str({input_shapes[i]}) << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "Snippets=" << snippets_enabled << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MatMulPerf::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    bool snippets_enabled;
    std::tie(input_shapes, elem_types, ref_num_nodes, ref_num_subgraphs, snippets_enabled, targetDevice) = this->GetParam();
    if (!snippets_enabled)
        ref_num_subgraphs = 0;
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::MatMulFunction(input_shapes, elem_types);
    function = f.getOriginal();
    if (!snippets_enabled) {
        ref_num_subgraphs = 0;
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::DISABLE});
    } else if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

std::string MHAWOTransposePerf::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParamsPerf> obj) {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    bool snippets_enabled;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, withMul, num_nodes, num_subgraphs, snippets_enabled, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::partialShape2str({inputShapes[i]}) << "_";
    result << "Mul=" << withMul << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "Snippets=" << snippets_enabled << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MHAWOTransposePerf::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    bool snippets_enabled;
    std::tie(inputShapes, withMul, ref_num_nodes, ref_num_subgraphs, snippets_enabled, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHAWOTransposeFunction(inputDynamicShapes, withMul);
    function = f.getOriginal();
    if (!snippets_enabled) {
        ref_num_subgraphs = 0;
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::DISABLE});
    } else if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHAWOTransposePerf::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();
    for (int i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), targetInputStaticShapes[i], 1.0f, 0.5f);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}

TEST_P(MatMulPerf, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
//    validateNumSubgraphs();
}

TEST_P(MHAWOTransposePerf, CompareWithRefImpl) {
    run();
//    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
