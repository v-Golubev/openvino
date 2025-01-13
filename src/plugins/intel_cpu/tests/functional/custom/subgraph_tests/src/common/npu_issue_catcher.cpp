// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/opsets/opset16.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class NPUIssueCatcherTest : virtual public SubgraphBaseStaticTest {
public:
    void BuildGraph() {
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration.insert({"DYNAMIC_QUANTIZATION_GROUP_SIZE", "0"});

        auto param = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape({1, 1, 1024}));
        const size_t up_to = 15;
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(ov::element::u8,
                                                                      {512, 1024},
                                                                      ov::test::utils::InputGenerateData(1, up_to));
        auto weights = std::make_shared<ov::opset10::Constant>(weights_tensor);
        auto convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f32);
        auto shift_const_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                                          {1, 1},
                                                                          ov::test::utils::InputGenerateData(1, up_to));
        auto shift = std::make_shared<ov::op::v0::Constant>(shift_const_tensor);
        auto sub = std::make_shared<ov::opset10::Subtract>(convert, shift);
        auto scale_const_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(ov::element::f32,
                                                                                            {1, 1},
                                                                                            0.001f,
                                                                                            0.01f,
                                                                                            1);
        auto scale = std::make_shared<ov::opset10::Constant>(scale_const_tensor);
        auto mul = std::make_shared<ov::opset10::Multiply>(sub, scale);
        auto matmul = std::make_shared<ov::opset10::MatMul>(param, mul, false, true);
        function = std::make_shared<ov::Model>(matmul, ov::ParameterVector{param}, "NPUIssueCatcher");
    }
};

namespace {
TEST_F(NPUIssueCatcherTest, smoke_NPUIssueCatcherTest_CPU) {
    BuildGraph();
    run();
}
}  // namespace
}  // namespace test
}  // namespace ov
