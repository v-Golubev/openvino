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
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/mat_mul.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class MatMullTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precisionBeforeDequantization2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precisionBeforeDequantization2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
        ngraph::element::Type precisionBeforeOperation1;
        ngraph::element::Type precisionBeforeOperation2;
        ngraph::builder::subgraph::DequantizationOperations result;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.dequantization1 << "_" << actual.dequantization2;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" <<
        expected.precisionBeforeDequantization1 << "_" <<
        expected.dequantization1 << "_" <<
        expected.precisionBeforeDequantization2 << "_" <<
        expected.dequantization2 << "_" <<
        expected.precisionBeforeOperation1 << "_" <<
        expected.precisionBeforeOperation2 << "_" <<
        expected.result;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.params.updatePrecisions << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    std::pair<ngraph::Shape, ngraph::Shape>,
    MatMullTransformationTestValues> MatMulTransformationParams;

class MatMulTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const std::pair<ngraph::Shape, ngraph::Shape> shapes = std::get<1>(GetParam());
        const MatMullTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(
            shapes.first,
            testValues.actual.precisionBeforeDequantization1,
            testValues.actual.dequantization1,
            shapes.second,
            testValues.actual.precisionBeforeDequantization2,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getReference(
            shapes.first,
            testValues.expected.precisionBeforeDequantization1,
            testValues.expected.dequantization1,
            shapes.second,
            testValues.expected.precisionBeforeDequantization2,
            testValues.expected.dequantization2,
            testValues.expected.result);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        std::pair<ngraph::Shape, ngraph::Shape> shapes;
        MatMullTransformationTestValues testValues;
        std::tie(precision, shapes, testValues) = obj.param;

        std::stringstream ss;
        ss << precision << "_" << shapes.first << "_" << shapes.second << "_" << testValues;
        return ss.str();
    }
};

TEST_P(MatMulTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape>> shapes = {
    { { 1, 16, 384, 64 }, { 1, 16, 64, 384 } }
};

const std::vector<bool> updatePrecisions = { true, false };

std::vector<MatMullTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8U8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.03f } },
        },
        {
            ngraph::element::u8,
            { {}, {}, {} },
            ngraph::element::i8,
            { {}, {}, {} },
            ngraph::element::u8,
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.03f } },
        }
    },
    {
        LayerTransformation::createParamsU8U8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.03f } },
        },
        {
            ngraph::element::u8,
            { {}, {}, {} },
            ngraph::element::u8,
            { {}, {}, {} },
            ngraph::element::u8,
            ngraph::element::u8,
            { {}, {}, { 0.02f * 0.03f } },
        }
    },
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(true),
        {
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.02f } },
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.03f } },
        },
        {
            ngraph::element::i8,
            { {}, {}, {} },
            ngraph::element::i8,
            { {}, {}, {} },
            ngraph::element::i8,
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.03f } },
        }
    },
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            { {}, {}, { 0.02f } },
            ngraph::element::f32,
            { {}, {}, { 0.03f } },
        },
        {
            ngraph::element::f32,
            { {}, {}, {} },
            ngraph::element::f32,
            { {}, {}, {} },
            ngraph::element::f32,
            ngraph::element::f32,
            { {}, {}, { 0.02f * 0.03f } },
        }
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MatMulTransformation::getTestCaseName);

} // namespace
