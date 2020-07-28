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
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/concat.hpp>
#include <transformations/low_precision/concat_multi_channels.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.fakeQuantize3 << "_" <<
        values.dequantizationOperations1 << "_" <<
        values.dequantizationOperations2;
}

class ConcatTransformationTestValues {
public:
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    bool,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithNeighborsTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const bool updatePrecisions = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        testValues.params.updatePrecisions = updatePrecisions;
        if (!updatePrecisions) {
            testValues.result.fakeQuantize1.outputPrecision = testValues.actual.fakeQuantize1.outputPrecision;
            testValues.result.fakeQuantize2.outputPrecision = testValues.actual.fakeQuantize2.outputPrecision;
            testValues.result.fakeQuantize3.outputPrecision = testValues.actual.fakeQuantize3.outputPrecision;
        }

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
            precision,
            testValues.inputShape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3);

        SimpleLowPrecisionTransformer transform;
        if (testValues.multiChannels) {
            transform.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        } else {
            transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        }
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceWithNeighbors(
            precision,
            testValues.inputShape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.dequantizationOperations1,
            testValues.result.dequantizationOperations2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const bool updatePrecision = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, testValues.inputShape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            (updatePrecision ? "updatePrecision_" : "notUpdatePrecision_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithNeighborsTransformation, CompareFunctions) {
    const ConcatTransformationTestValues testValues = std::get<2>(GetParam());

    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<bool> updatePrecisions = { true, false };

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {128.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {85.f}, ngraph::element::u8 },
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat multi channels
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        }
    },
    // U8: concat multi channels with subtract
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { ngraph::element::f32, {{ 0.f, 0.f, 0.f, -255.f, -255.f, -255.f }}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, { -255.f }, { 0.005f } }
        }
    },
    // I8: concat
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8 },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-64}, {64.f}, ngraph::element::i8 },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-43}, {42.f}, ngraph::element::i8 },
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // I8: concat multi channels
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsI8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8 },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f}, ngraph::element::i8 },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f}, ngraph::element::i8 },
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        }
    },
    // mixed: U8 + I8: concat multi channels
    {
        { 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8 },
            { ngraph::element::f32, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } },
            { ngraph::element::f32, { 128.f }, { 0.01f } }
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ConcatWithNeighborsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(testValues)),
    ConcatWithNeighborsTransformation::getTestCaseName);
}  // namespace
