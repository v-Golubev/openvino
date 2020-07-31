// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvolutionTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class ConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionTransformationTestValues> {
public:
    void SetUp() override {
        const ConvolutionTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.actual.weightsPrecision,
            testValues.actual.weightsValues,
            testValues.actual.fakeQuantizeOnWeights);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::getReference(
            testValues.inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            testValues.expected.weightsPrecision,
            testValues.expected.weightsValues,
            testValues.expected.fakeQuantizeOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationTestValues> obj) {
        const ConvolutionTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            toString(testValues.params) << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.precisionBeforeDequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.precisionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter;
        return result.str();
    }
};

TEST_P(ConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ConvolutionTransformationTestValues> testParams = {
// with zero point
{
    Shape{1, 32, 72, 48},
    LayerTransformation::createParamsU8I8(),
    //ActualValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {128.f}, {0.02f}},
        ngraph::element::f32,
        {2.f},
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
    },
    //ExpectedValues
    {
        // onActivation
        ngraph::element::u8,
        {{}, {128.f}, {}},
        // onWeights
        ngraph::element::i8,
        { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
        {},
        // afterOperation
        ngraph::element::f32,
        {{}, {}, {{0.0002f}, ngraph::element::f32, {1, 1, 1}}}, // 0.0002 = 0.02 (on data) * 0.01 (on weights)
    }
},
// without zero point
{
    Shape{1, 32, 72, 48},
    LayerTransformation::createParamsU8I8(),
    // ActualValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {}, {0.02f}},
        ngraph::element::f32,
        { 2.f },
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
    },
    // ExpectedValues
    {
        ngraph::element::u8,
        {{}, {}, {}},
        ngraph::element::i8,
        { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
        {},
        ngraph::element::f32,
        {{}, {}, {{0.0002f}, ngraph::element::f32, {1, 1, 1}}}, // 0.0002 = 0.02 (on data) * 0.01 (on weights)
    }
},
// u8 isn't supported
{
    Shape{1, 32, 72, 48},
    LayerTransformation::createParamsI8I8(),
    // ActualValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {128.f}, {0.02f}},
        ngraph::element::f32,
        { 2.f },
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
    },
    // ExpectedValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {128.f}, {0.02f}},
        ngraph::element::f32,
        { 2.f },
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} },
        ngraph::element::f32,
        {{}, {}, {}},
    }
},
// with zero point, updatePrecision = false
{
    Shape{1, 32, 72, 48},
    LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
    //ActualValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {128.f}, {0.02f}},
        ngraph::element::f32,
        {2.f},
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
    },
    //ExpectedValues
    {
        // onActivation
        ngraph::element::u8,
        {{}, {128.f}, {}},
        // onWeights
        ngraph::element::f32,
        { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
        {},
        // afterOperation
        ngraph::element::f32,
        {{}, {}, {{0.0002f}, ngraph::element::f32, {1, 1, 1}}}, // 0.0002 = 0.02 (on data) * 0.01 (on weights)
    }
},
// without zero point, updatePrecision = false
{
    Shape{1, 32, 72, 48},
    LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
    // ActualValues
    {
        ngraph::element::u8,
        {{ngraph::element::f32}, {}, {0.02f}},
        ngraph::element::f32,
        { 2.f },
        { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
    },
    // ExpectedValues
    {
        ngraph::element::u8,
        {{}, {}, {}},
        ngraph::element::f32,
        { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
        {},
        ngraph::element::f32,
        {{}, {}, {{0.0002f}, ngraph::element::f32, {1, 1, 1}}}, // 0.0002 = 0.02 (on data) * 0.01 (on weights)
    }
},
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ConvolutionTransformation,
    ::testing::ValuesIn(testParams),
    ConvolutionTransformation::getTestCaseName);
