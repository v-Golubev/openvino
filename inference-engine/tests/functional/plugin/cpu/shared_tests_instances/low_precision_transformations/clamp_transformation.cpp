// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/clamp_transformation.hpp"


using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    // LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<LayerTestsDefinitions::ClampTransformationParam> params{
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        { 0.0 },
        { 127.0 }
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { 0.f }, { 255.f } },
        { 0.0 },
        { 255.0 }
    },
    {
        {
            256ul, ngraph::Shape
            { 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        { 0.0 },
        { 128.0 }
    }
};

INSTANTIATE_TEST_CASE_P(LPT, ClampTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versions),
        ::testing::ValuesIn(params)),
    ClampTransformation::getTestCaseName);

}  // namespace
