// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "openvino/openvino.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;

using namespace ov::test;

namespace LayerTestsDefinitions {

class CustomTest : public SubgraphBaseTest {
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        InputShape inputShapes{{}, {{3, 400, 196, 80}}};
        init_input_shapes({inputShapes});

        ov::Core core;
        function = core.read_model("/home/vgolubev/models/segmentation_any/subgraph.xml");
    }
};

namespace  {
TEST_F(CustomTest, smoke_CustomTest_CPU) {
    run();
}
} // namespace
} // namespace LayerTestsDefinitions