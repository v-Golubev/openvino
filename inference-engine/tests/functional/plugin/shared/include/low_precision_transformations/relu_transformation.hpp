// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    LayerTestsUtils::LayerTransformation::LptVersion,
    ngraph::builder::subgraph::FakeQuantizeOnData> ReluTransformationParams;

class ReluTransformation :
    public testing::WithParamInterface<ReluTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReluTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
