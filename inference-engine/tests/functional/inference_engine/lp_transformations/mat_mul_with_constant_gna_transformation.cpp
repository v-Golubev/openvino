// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/mat_mul.hpp>
#include <transformations/low_precision/fake_quantize.hpp>

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
        ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        ngraph::element::Type precisionAfterDequantization;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.fqOnData << "_" << actual.fqOnWeights;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" <<
        expected.dequantizationBefore << "_" <<
        expected.dequantizationAfter << "_" <<
        expected.precisionAfterDequantization;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    MatMullTransformationTestValues> MatMulTransformationParams;

class MatMulWithConstantGNATransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const MatMullTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getGNAOriginal(
            precision,
            shape,
            testValues.actual.fqOnData,
            testValues.actual.weights,
            testValues.actual.fqOnWeights);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transformer.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getGNAReference(
            precision,
            shape,
            testValues.expected.fqOnData,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.weights,
            testValues.expected.fqOnWeights,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shapes;
        MatMullTransformationTestValues testValues;
        std::tie(precision, shapes, testValues) = obj.param;

        std::stringstream ss;
        ss << precision << "_" << shapes << "_" << testValues;
        return ss.str();
    }
};

TEST_P(MatMulWithConstantGNATransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::i32,
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 2048 },
    { 4, 2048 }
};

std::vector<MatMullTransformationTestValues> testValues = {
    // U16 & I16
    {
        LayerTransformation::createParamsU16I16().setSupportAsymmetricQuantization(false),
        {
            { 16384, ngraph::Shape({}),  {0.f}, {16383.f}, {0.f}, {16383.f * 2.f} },
            ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 8191)),
            { 16383, ngraph::Shape({}),  {-8191.f}, {8191.f}, {-8191.f * 3.f}, {8191.f * 3.f} },
        },
        {
            { 16384, ngraph::Shape({}),  {0.f}, {16383.f}, {0.f}, {16383.f} },
            ngraph::element::u16,
            {},
            ngraph::op::Constant::create(ngraph::element::i16, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 8191)),
            {},
            ngraph::element::i32,
            {{}, {}, { 6.f } }, // 2(on data) * 3(on weights)
            ngraph::element::i32,
        }
    },
    // U16 & I16
    {
        LayerTransformation::createParamsU16I16().setSupportAsymmetricQuantization(false),
        {
            { 16384, ngraph::Shape({}),  {0.f}, {16383.f}, {0.f}, {16383.f * 2.f} },
            ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 8200)),
            { 16383, ngraph::Shape({}),  {0.f}, {16382.f}, {-8191.f}, {8191.f} },
        },
        {
            { 16384, ngraph::Shape({}),  {0.f}, {16383.f}, {0.f}, {16383.f} },
            ngraph::element::u16,
            {},
            // 8200 - 8191(FakeQuantize)
            ngraph::op::Constant::create(ngraph::element::i16, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 9)),
            {},
            ngraph::element::i32,
            {{}, {}, { 2.f } },
            ngraph::element::i32,
        }
    },
    // U16 & I16
    {
        LayerTransformation::createParamsU16I16().setSupportAsymmetricQuantization(false),
        {
            { 16383, ngraph::Shape({}),  {-8191.f}, {8191.f}, {-8191.f}, {8191.f} },
            ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 8191)),
            { 16383, ngraph::Shape({}),  {-8191.f}, {8191.f}, {-8191.f}, {8191.f} },
        },
        {
            { 16383, ngraph::Shape({}),  {-8191.f}, {8191.f}, {0.f}, {16382.f} },
            ngraph::element::u16,
            {{ngraph::element::i32}, {8191.f}, {1}},
            ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{ 2048, 1000 }, std::vector<float>(2048 * 1000, 8191)),
            { 16383, ngraph::Shape({}),  {-8191.f}, {8191.f}, {-8191.f}, {8191.f} },
            ngraph::element::i32,
            {{}, {}, {} },
            ngraph::element::i32,
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulWithConstantGNATransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantGNATransformation::getTestCaseName);

} // namespace
