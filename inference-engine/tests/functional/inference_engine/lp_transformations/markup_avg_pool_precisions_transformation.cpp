// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include <low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp>
#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>
#include <low_precision/rt_info/precisions_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <low_precision/rt_info/shared_value_attribute.hpp>

#include <low_precision/create_attribute.hpp>
#include <low_precision/create_precisions_dependent_attribute.hpp>
#include <low_precision/propagate_through_precision_preserved.hpp>
#include <low_precision/propagate_to_input.hpp>
#include <low_precision/update_shared_precision_preserved.hpp>

#include <low_precision/low_precision.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precision_preserved.hpp>
#include <low_precision/propagate_precisions.hpp>
#include <low_precision/avg_pool.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/align_quantization_parameters.hpp>

// cleanup transformations
#include "low_precision/fake_quantize.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/markup_avg_pool_precisions_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;

class MarkupAvgPoolPrecisionsTransformationTestValues {
public:
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool, // additional FakeQuantize After
    std::string, // additional layer before FQ
    MarkupAvgPoolPrecisionsTransformationTestValues> MarkupAvgPoolPrecisionsTransformationParams;

class MarkupAvgPoolPrecisionsTransformation : public LayerTransformation, public testing::WithParamInterface<MarkupAvgPoolPrecisionsTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();
        actualFunction = ngraph::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getOriginal(
            precision,
            testValues.actual.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.actual.dequantization,
            1,
            0);

//#define VISUALIZE_TREE
#ifndef VISUALIZE_TREE
        ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer pass;
        pass.run_on_function(actualFunction);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

        ngraph::pass::Manager manager;

        manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
        manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
        manager.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();

        std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
        common->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>();
        common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();

        std::shared_ptr<ngraph::pass::GraphRewrite> cleanup = manager.register_pass<ngraph::pass::GraphRewrite>();
        cleanup->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>();
        cleanup->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
        cleanup->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();

        manager.run_passes(actualFunction);
#else
        ngraph::pass::VisualizeTree("~/projects/temp/test.actual").run_on_function(actualFunction);

        ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer pass;
        pass.run_on_function(actualFunction);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });
        ngraph::pass::Manager manager1;
        manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager1.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming1.svg").run_on_function(actualFunction);

        //ngraph::pass::Manager manager2;
        //manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
        //manager2.run_passes(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager;
            //manager.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>();
            std::shared_ptr<ngraph::pass::GraphRewrite> markupAvgPoolPrecision = manager.register_pass<ngraph::pass::GraphRewrite>();
            markupAvgPoolPrecision->add_matcher<low_precision::CreatePrecisionsDependentAttribute<AvgPoolPrecisionPreservedAttribute, opset1::AvgPool>>();
            markupAvgPoolPrecision->add_matcher<low_precision::PropagateThroughPrecisionPreserved<AvgPoolPrecisionPreservedAttribute>>();
            markupAvgPoolPrecision->add_matcher<low_precision::UpdateSharedPrecisionPreserved<AvgPoolPrecisionPreservedAttribute>>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming2.svg").run_on_function(actualFunction);
        }

        //ngraph::pass::Manager manager3;
        //manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        //manager3.run_passes(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager;
            //manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
            std::shared_ptr<ngraph::pass::GraphRewrite> precisionsPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
            precisionsPropagation->add_matcher<low_precision::CreateAttribute<PrecisionsAttribute, opset1::FakeQuantize>>(AttributeSource::OutputPort);
            precisionsPropagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<PrecisionsAttribute>>();
            precisionsPropagation->add_matcher<low_precision::PropagateToInput<PrecisionsAttribute>>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming3.svg").run_on_function(actualFunction);
        }

        ngraph::pass::Manager manager4;
        manager4.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
        manager4.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming4.svg").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>();
            common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
            common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
            common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();

            std::shared_ptr<ngraph::pass::GraphRewrite> cleanup = manager.register_pass<ngraph::pass::GraphRewrite>();
            cleanup->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>();
            cleanup->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            cleanup->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();

            manager.run_passes(actualFunction);
        }

        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transformed.svg").run_on_function(actualFunction);
#endif

        referenceFunction = ngraph::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);

        // ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.reference.svg").run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MarkupAvgPoolPrecisionsTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, shape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(MarkupAvgPoolPrecisionsTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    const auto avgPoolOperations = LayerTransformation::get<opset1::AvgPool>(actualFunction);
    ASSERT_EQ(1ul, avgPoolOperations.size()) << "unexpected avgPoolOperations size: " << avgPoolOperations.size();

    {
        auto avgPoolPrecisioinPreservedAttribute = ngraph::pass::low_precision::getAttribute<AvgPoolPrecisionPreservedAttributePtr>(
                *avgPoolOperations.begin());
        ASSERT_NE(nullptr, avgPoolPrecisioinPreservedAttribute);
        ASSERT_EQ(true, avgPoolPrecisioinPreservedAttribute->get()->sharedValue->value);
    }

    const auto precisionPreserved = LayerTransformation::get<opset1::MaxPool>(actualFunction);
    ASSERT_TRUE(checkIfAttributesAreTheSame<std::shared_ptr<AvgPoolPrecisionPreservedAttribute>>(precisionPreserved)) <<
        "AvgPoolPrecisionPreservedAttribute are not the same";

    //auto res = compare_functions(referenceFunction, actualFunction, true, true);
    //ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<std::string> additionalLayer = {
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    //true,
    false
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 }
};

const std::vector<MarkupAvgPoolPrecisionsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
    //// U8 without subtract
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        {{ngraph::element::f32}, {}, {0.02f}}
    //    },
    //    {
    //        ngraph::element::u8,
    //        {},
    //        ngraph::element::f32,
    //        {{}, {}, {0.02f}}
    //    }
    //},
    //// U8 per channel quantization with different values
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        {
    //            {ngraph::element::f32},
    //            {{128.f, 0.f, 128.f / 2}},
    //            {{3.f, 1.f, 2.f}}
    //        }
    //    },
    //    {
    //        ngraph::element::u8,
    //        {{}, {}, {}},
    //        ngraph::element::f32,
    //        {
    //            {},
    //            {{128.f, 0.f, 128.f / 2}},
    //            {{3.f, 1.f, 2.f}}
    //        },
    //    }
    //},
    //// U8 per channel quantization with the same values
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        {
    //            {ngraph::element::f32},
    //            {{128.f, 128.f, 128.f}},
    //            {{3.f, 3.f, 3.f}}
    //        }
    //    },
    //    {
    //        ngraph::element::u8,
    //        {{}, {}, {}},
    //        ngraph::element::f32,
    //        {
    //            {},
    //            {{128.f, 128.f, 128.f}},
    //            {{3.f, 3.f, 3.f}}
    //        },
    //    }
    //},
    //// U8 without dequantization
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        {}
    //    },
    //    {
    //        ngraph::element::u8,
    //        {},
    //        ngraph::element::u8,
    //        {}
    //    }
    //},
    //// U8 not update precisions
    //{
    //    LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
    //    {
    //        ngraph::element::f32,
    //        {{}, {128.f}, {0.02f}}
    //    },
    //    {
    //        ngraph::element::f32,
    //        {},
    //        ngraph::element::f32,
    //        {{}, {128.f}, {0.02f}}
    //    }
    //},
    //// I8 per tensor quantization
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    {
    //        ngraph::element::i8,
    //        {{ngraph::element::f32}, {128.f}, {0.02f}}
    //    },
    //    {
    //        ngraph::element::i8,
    //        {},
    //        ngraph::element::f32,
    //        {{}, {128.f}, {0.02f}}
    //    }
    //},
    //// I8 without subtract
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    {
    //        ngraph::element::i8,
    //        {{ngraph::element::f32}, {}, {0.02f}}
    //    },
    //    {
    //        ngraph::element::i8,
    //        {},
    //        ngraph::element::f32,
    //        {{}, {}, {0.02f}}
    //    }
    //},
    //// I8 per channel quantization with different values
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    {
    //        ngraph::element::i8,
    //        {
    //            {ngraph::element::f32},
    //            {{64.f, 0.f, 32.f}},
    //            {{3.f, 1.f, 2.f}}
    //        }
    //    },
    //    {
    //        ngraph::element::i8,
    //        {{}, {}, {}},
    //        ngraph::element::f32,
    //        {
    //            {},
    //            {{64.f, 0.f, 32.f}},
    //            {{3.f, 1.f, 2.f}}
    //        },
    //    }
    //},
    //// I8 per channel quantization with the same values
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    {
    //        ngraph::element::i8,
    //        {
    //            {ngraph::element::f32},
    //            {{64.f, 64.f, 64.f}},
    //            {{3.f, 3.f, 3.f}}
    //        }
    //    },
    //    {
    //        ngraph::element::i8,
    //        {{}, {}, {}},
    //        ngraph::element::f32,
    //        {
    //            {},
    //            {{64.f, 64.f, 64.f}},
    //            {{3.f, 3.f, 3.f}}
    //        },
    //    }
    //},
    //// I8 without dequantization
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    {
    //        ngraph::element::i8,
    //        {}
    //    },
    //    {
    //        ngraph::element::i8,
    //        {},
    //        ngraph::element::i8,
    //        {}
    //    }
    //},
    //// I8 not update precisions
    //{
    //    LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
    //    {
    //        ngraph::element::f32,
    //        {{}, {128.f}, {0.02f}}
    //    },
    //    {
    //        ngraph::element::f32,
    //        {},
    //        ngraph::element::f32,
    //        {{}, {128.f}, {0.02f}}
    //    }
    //},
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    MarkupAvgPoolPrecisionsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(addFQ),
        ::testing::ValuesIn(additionalLayer),
        ::testing::ValuesIn(testValues)),
    MarkupAvgPoolPrecisionsTransformation::getTestCaseName);
