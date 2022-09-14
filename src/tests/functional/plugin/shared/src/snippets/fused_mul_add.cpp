// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/fused_mul_add.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string FusedMulAdd::getTestCaseName(testing::TestParamInfo<ov::test::snippets::FusedMulAddParams> obj) {
    ov::Shape inputShapes0, inputShapes1, inputShapes2;
    size_t input_idx, num_nodes, num_subgraphs;
    std::string targetDevice;
    std::tie(inputShapes0, inputShapes1, inputShapes2, input_idx, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
    result << "IS[2]=" << CommonTestUtils::vec2str(inputShapes2) << "_";
    result << "inputIndex=" << input_idx << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void FusedMulAdd::SetUp() {
    ov::Shape inputShape0, inputShape1, inputShape2;
    size_t input_idx;
    std::tie(inputShape0, inputShape1, inputShape2, input_idx, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}, {{}, {inputShape1, }}});

    auto f = ov::test::snippets::EltwiseWithMulAddFunction({inputShape0, inputShape1, inputShape2}, input_idx);
    function = f.getOriginal();
}

TEST_P(FusedMulAdd, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
