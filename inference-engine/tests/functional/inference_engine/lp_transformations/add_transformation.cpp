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

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include <transformations/low_precision/add.hpp>
#include "ngraph_functions/low_precision_transformations/add_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class AddTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    AddActualValues actual;
    AddExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    AddTransformationTestValues> AddTransformationParams;

class AddTransformation : public LayerTransformation, public testing::WithParamInterface<AddTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool broadcast = std::get<2>(GetParam());
        const AddTransformationTestValues testParams = std::get<3>(GetParam());

        actualFunction = AddFunction::getOriginal(
            precision,
            shape,
            broadcast,
            testParams.transformationParams,
            testParams.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AddTransformation, ngraph::opset1::Add>(
            low_precision::LayerTransformation::Params(testParams.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = AddFunction::getReference(
            precision,
            shape,
            broadcast,
            testParams.transformationParams,
            testParams.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AddTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool broadcast;
        AddTransformationTestValues params;
        std::tie(precision, shape, broadcast, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, params.transformationParams) <<
            (broadcast ? "_broadcast_" : "") << params.actual << params.expected;
        return result.str();
    }
};

TEST_P(AddTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<bool> broadcastValues = {
    true,
    false
};

const std::vector<AddTransformationTestValues> addTransformationTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 7.f }, { 10.f }, ngraph::element::u8, { 3.f }, { 5.f } },
        { ngraph::element::u8, { 8.5f }, { 2.f }, ngraph::element::u8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 2.f }, { 10.f }, ngraph::element::u8, { }, { 5.f } },
        { ngraph::element::u8, { 2.f }, { 2.f }, ngraph::element::u8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, {  }, { 10.f }, ngraph::element::u8, { }, { 5.f } },
        { ngraph::element::u8, {  }, { 2.f }, ngraph::element::u8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 2.f }, {  }, ngraph::element::u8, { }, { 5.f } },
        { ngraph::element::u8, { 2.f }, { 0.2f }, ngraph::element::u8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 2.f }, {  }, ngraph::element::u8, { 3.f }, { 5.f } },
        { ngraph::element::u8, { 17.f }, { 0.2f }, ngraph::element::u8, { 5.f } }
    },

    // I8
    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::i8, { 7.f }, { 10.f }, ngraph::element::i8, { 3.f }, { 5.f } },
        { ngraph::element::i8, { 8.5f }, { 2.f }, ngraph::element::i8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::i8, { 2.f }, { 10.f }, ngraph::element::i8, { }, { 5.f } },
        { ngraph::element::i8, { 2.f }, { 2.f }, ngraph::element::i8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::i8, {  }, { 10.f }, ngraph::element::i8, { }, { 5.f } },
        { ngraph::element::i8, {  }, { 2.f }, ngraph::element::i8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::i8, { 2.f }, {  }, ngraph::element::i8, { }, { 5.f } },
        { ngraph::element::i8, { 2.f }, { 0.2f }, ngraph::element::i8, { 5.f } }
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::i8, { 2.f }, {  }, ngraph::element::i8, { 3.f }, { 5.f } },
        { ngraph::element::i8, { 17.f }, { 0.2f }, ngraph::element::i8, { 5.f } }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(broadcastValues),
        ::testing::ValuesIn(addTransformationTestValues)),
    AddTransformation::getTestCaseName);
