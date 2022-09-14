// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fused_mul_add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

const std::vector<size_t> input_idxes = {0, 1};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FusedMulAdd, FusedMulAdd,
                        ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 64, 10, 10}),
                             ::testing::Values(ov::Shape {1, 64, 10,  1}),
                             ::testing::Values(ov::Shape {1, 64, 10,  1}),
                             ::testing::ValuesIn(input_idxes),
                             ::testing::Values(4), // 3 sin after inputs and subgraph with fma
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         FusedMulAdd::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov