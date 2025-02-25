// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/exp.hpp"
#include "subgraph_simple.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Exp::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ExpParams> obj) {
    ov::test::InputShape inputShapes0;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes0.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Exp::SetUp() {
    ov::test::InputShape inputShape0;
    ov::element::Type type;
    std::tie(inputShape0, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0});
    auto f = ov::test::snippets::ExpFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void SubExp::SetUp() {
    ov::test::InputShape inputShape0;
    ov::element::Type type;
    std::tie(inputShape0, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    ov::test::InputShape inputShape1;
    const auto input_shape_0 = inputShape0.second.back();
    std::cout << ov::PartialShape(input_shape_0) << std::endl;
    const auto input_shape_1 = ov::Shape{1, 1, input_shape_0[input_shape_0.size() - 2], 1};
    init_input_shapes({{{}, {input_shape_0}}, {{}, {input_shape_1}}});

    auto data0 = std::make_shared<op::v0::Parameter>(type, inputDynamicShapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(type, inputDynamicShapes[1]);
    auto sub = std::make_shared<ov::op::v1::Subtract>(data0, data1);
    auto exp = std::make_shared<op::v0::Exp>(sub);
    function = std::make_shared<ov::Model>(NodeVector{exp}, ParameterVector{data0, data1});

    setInferenceType(type);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void SubExp::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const bool use_default_generate = true;
    if (use_default_generate) {
        ov::test::SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        return;
    }

    auto generate = [=](size_t i, float val) {
        const auto& dataShape = targetInputStaticShapes[i];
        auto tensor = ov::Tensor{ov::element::f32, dataShape};
        auto begin = tensor.data<float>();
        std::fill(begin, begin + ov::shape_size(dataShape), val);
        return tensor;
    };

    inputs.clear();
    const auto& dataShape = targetInputStaticShapes[0];
    const auto funcInput0 = function->inputs()[0];
    const auto funcInput1 = function->inputs()[1];
    inputs.insert({funcInput0.get_node_shared_ptr(), generate(0, std::numeric_limits<float>::infinity())});
    inputs.insert({funcInput1.get_node_shared_ptr(), generate(1, 0.f)});
}

void ExpReciprocal::SetUp() {
    ov::test::InputShape inputShape0;
    ov::element::Type type;
    std::tie(inputShape0, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0});
    auto f = ov::test::snippets::ExpReciprocalFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}


TEST_P(Exp, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ExpReciprocal, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(SubExp, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
