// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/split.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/split_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class SplitTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    ngraph::builder::subgraph::SplitFunction::ActualValues actual;
    ngraph::builder::subgraph::SplitFunction::ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    SplitTransformationTestValues> SplitTransformationParams;

class SplitTransformation : public LayerTransformation, public testing::WithParamInterface<SplitTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecisions = std::get<2>(GetParam());
        const SplitTransformationTestValues testValues = std::get<3>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testValues.transformationParams).
            setUpdatePrecisions(updatePrecisions);

        actualFunction = ngraph::builder::subgraph::SplitFunction::getOriginal(
            precision,
            shape,
            updatePrecisions,
            testValues.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::SplitTransformation, ngraph::opset1::Split>(testValues.transformationParams);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::SplitFunction::getReference(
            precision,
            shape,
            updatePrecisions,
            testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SplitTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const bool updatePrecisions = std::get<2>(obj.param);
        SplitTransformationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.transformationParams.setUpdatePrecisions(updatePrecisions)) <<
            testValues.actual << testValues.expected;
        return result.str();
    }
};

TEST_P(SplitTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 24, 24 }
};

const std::vector<bool> updatePrecision = {
    true,
    false
};

const std::vector<SplitTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { }, { 128.f },
            { }, { 3.f },
            2, 8
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { }, { { 128.f }, { 128.f }, { 128.f }, { 128.f }, { 128.f }, { 128.f }, { 128.f }, { 128.f } },
            { }, { { 3.f }, { 3.f }, { 3.f }, { 3.f }, { 3.f }, { 3.f }, { 3.f }, { 3.f } },
            2, 8
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        // Actualvalues
        {
            ngraph::element::i8,
            { 1, 3, 1, 1 }, { 11.f, 22.f, 33.f },
            { 1, 3, 1, 1 }, { 1.f, 2.f, 3.f },
            1, 3
        },
        // Expectedvalues
        {
            ngraph::element::i8,
            { 1, 1, 1, 1 }, { { 11.f }, { 22.f }, { 33.f } },
            { 1, 1, 1, 1 }, { { 1.f }, { 2.f }, { 3.f } },
            1, 3
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        // Actualvalues
        {
            ngraph::element::u8,
            { 1, 3, 1, 1 }, { 11.f, 22.f, 33.f },
            { 1, 3, 1, 1 }, { 1.f, 2.f, 3.f },
            -1, 3
        },
        // Expectedvalues
        {
            ngraph::element::u8,
            { 1, 3, 1, 1 }, { { 11.f, 22.f, 33.f }, { 11.f, 22.f, 33.f }, { 11.f, 22.f, 33.f }, { 11.f, 22.f, 33.f } },
            { 1, 3, 1, 1 }, { { 1.f, 2.f, 3.f }, { 1.f, 2.f, 3.f }, { 1.f, 2.f, 3.f }, { 1.f, 2.f, 3.f } },
            -1, 3
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        // Actualvalues
        {
            ngraph::element::i8,
            { }, { 11.f },
            { }, { 1.f },
            2, 2
        },
        // Expectedvalues
        {
            ngraph::element::i8,
            { }, { { 11.f }, { 11.f } },
            { }, { { 1.f }, { 1.f } },
            2, 2
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        // Actualvalues
        {
            ngraph::element::i8,
            { }, { },
            { }, { 1.f },
            2, 2
        },
        // Expectedvalues
        {
            ngraph::element::i8,
            { }, { },
            { }, { { 1.f }, { 1.f } },
            2, 2
        }
    }
};
INSTANTIATE_TEST_CASE_P(
    LPT,
    SplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecision),
        ::testing::ValuesIn(testValues)),
    SplitTransformation::getTestCaseName);
