// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/exp.hpp"
#include "common_test_utils/test_constants.hpp"

#include <shared_test_classes/base/benchmark.hpp>

namespace ov {
namespace test {
namespace snippets {


namespace {
// ===================================Exp=========================================================//
std::vector<ov::test::InputShape> inShapes{
    {PartialShape{}, {{1, 1, 2, 4096}}},
    // {PartialShape{-1, -1, -1}, {{1, 32, 128}, {1, 32, 30}, {1, 32, 1}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Exp, Exp,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(1), // Exp
                            ::testing::Values(1),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Exp::getTestCaseName);


struct SubExpBenchmarkTest : ov::test::BenchmarkLayerTest<SubExp> {};

TEST_P(SubExpBenchmarkTest, SubExp_Benchmark) {
    run_benchmark("Subgraph", std::chrono::milliseconds(2000), 10000);
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SubExp_Benchmark,
                         SubExpBenchmarkTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(1),  // Exp
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Exp::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExpReciprocal, ExpReciprocal,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(1), // Exp
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Exp::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov