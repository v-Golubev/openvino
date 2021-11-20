// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::test;

const std::vector<ElementType> netPrecisions{
    ElementType::f32
};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

namespace LRN2D {
const std::vector<std::vector<int64_t>> axes = {{1}};

const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{10, 16}}},
    InputShape{
        {-1, -1},
        {{10, 16}, {5, 12}, {3, 17}}
    },
    InputShape{
        {{3, 10}, {12, 17}},
        {{10, 16}, {5, 12}, {3, 17}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_2D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN2D

namespace LRN3D {
const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{6, 10, 16}}},
    InputShape{
        {-1, -1, -1},
        {{6, 10, 16}, {1, 5, 12}, {12, 3, 17}}
    },
    InputShape{
        {{1, 12}, {3, 10}, {12, 17}},
        {{6, 10, 16}, {1, 5, 12}, {12, 3, 17}}
    },
};

const std::vector<std::vector<int64_t>> axes = {{1}, {2}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_3D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN3D

namespace LRN4D {
const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{10, 10, 3, 8}}},
    InputShape{
        {-1, -1, -1, -1},
        {{10, 10, 3, 8}, {8, 8, 3, 6}, {5, 10, 5, 10}}
    },
    InputShape{
        {{5, 10}, {8, 10}, {3, 5}, {6, 10}},
        {{10, 10, 3, 8}, {8, 8, 3, 6}, {5, 10, 5, 10}}
    },
};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}, {3, 2}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_4D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN4D

namespace LRN5D {
const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{1, 10, 10, 3, 8}}},
    InputShape{
        {-1, -1, -1, -1, -1},
        {{1, 10, 10, 3, 8}, {2, 8, 8, 3, 6}, {3, 5, 10, 5, 10}}
    },
    InputShape{
        {{1, 3}, {5, 10}, {8, 10}, {3, 5}, {6, 10}},
        {{1, 10, 10, 3, 8}, {2, 8, 8, 3, 6}, {3, 5, 10, 5, 10}}
    },
};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3, 4}, {4, 2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_5D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);
} // namespace LRN5D
