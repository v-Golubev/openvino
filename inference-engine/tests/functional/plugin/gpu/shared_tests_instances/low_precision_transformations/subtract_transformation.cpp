// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/multiply_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::details::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

//INSTANTIATE_TEST_CASE_P(DISABLED_LPT, MultiplyTransformation,
//    ::testing::Combine(
//        ::testing::ValuesIn(netPrecisions),
//        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
//        ::testing::Values(CommonTestUtils::DEVICE_GPU),
//        ::testing::ValuesIn(trasformationParamValues),
//        ::testing::ValuesIn(versions)),
//    MultiplyTransformation::getTestCaseName);
}  // namespace




