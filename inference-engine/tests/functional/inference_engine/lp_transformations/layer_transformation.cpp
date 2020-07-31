// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/split.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/split_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph::pass;

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8U8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::u8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsI8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::i8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8AndI8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8, ngraph::element::i8 },
        { ngraph::element::i8 });
}

inline std::ostream& operator<<(std::ostream& os,
    const std::vector<ngraph::element::Type>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric_" : "symmetric_") <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        "onActivations:" << params.precisionsOnActivations << "_" <<
        "onWeights:" << params.precisionsOnWeights << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

void LayerTransformation::transform(std::shared_ptr<ngraph::Function> function) {
    ngraph::pass::low_precision::LowPrecisionTransformations transformations = ngraph::pass::low_precision::LowPrecisionTransformer::getAllTransformations();
    ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
    transformer.transform(function);
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type& type,
    const ngraph::Shape& shape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << type << "_" << shape << "_" << toString(params);
    return result.str();
}

namespace {
    using namespace testing;
    using namespace ngraph::pass;

    class BaseTransformationTestValues {
    public:
        class Actual {
        public:
            ngraph::element::Type precisionBeforeDequantization;
            ngraph::builder::subgraph::DequantizationOperations dequantization;
        };

        class Expected {
        public:
            ngraph::element::Type precisionBeforeDequantization;
            ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
            ngraph::element::Type precisionAfterOperation;
            ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        };

        ngraph::pass::low_precision::LayerTransformation::Params params;
        Actual actual;
        Expected expected;
    };

    class BaseTransformation : public LayerTransformation, public testing::WithParamInterface<BaseTransformationTestValues> {
    public:
        void SetUp() override {
            const BaseTransformationTestValues testValues = GetParam();

            actualFunction = ngraph::builder::subgraph::SplitFunction::getOriginal(
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization);

            SimpleLowPrecisionTransformer transformer;
            transformer.add<ngraph::pass::low_precision::SplitTransformation, ngraph::opset1::Split>(testValues.params);
            transformer.transform(actualFunction);

            referenceFunction = ngraph::builder::subgraph::SplitFunction::getReference(
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationBefore,
                testValues.expected.precisionAfterOperation,
                testValues.expected.dequantizationAfter);
        }

        static std::string getTestCaseName(testing::TestParamInfo<BaseTransformationTestValues> obj) {
            const BaseTransformationTestValues testValues = obj.param;

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

    TEST_P(BaseTransformation, CompareFunctions) {
        InitNodeInfo().run_on_function(actualFunction);
        actualFunction->validate_nodes_and_infer_types();

        auto res = compare_functions(referenceFunction, actualFunction, true);
        ASSERT_TRUE(res.first) << res.second;
    }

    const std::vector<BaseTransformationTestValues> testValues = {
        {
            LayerTransformation::createParamsU8U8(),
            {
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::u8,
                {{}, {}, {}},
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8U8(),
            {
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}},
                ngraph::element::f32,
                {{}, {}, {}}
            }
        },
        {
            LayerTransformation::createParamsU8U8().setUpdatePrecisions(false),
            {
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::f32,
                {{}, {}, {}},
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsI8I8(),
            {
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::i8,
                {{}, {}, {}},
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsI8I8(),
            {
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}},
                ngraph::element::f32,
                {{}, {}, {}}
            }
        },
        {
            LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
            {
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::f32,
                {{}, {}, {}},
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8(),
            {
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::u8,
                {{}, {}, {}},
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8(),
            {
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::i8,
                {{}, {}, {}},
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
            {
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::f32,
                {{}, {}, {}},
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8AndI8(),
            {
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::u8,
                {{}, {}, {}},
                ngraph::element::u8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8AndI8(),
            {
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::i8,
                {{}, {}, {}},
                ngraph::element::i8,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
        {
            LayerTransformation::createParamsU8I8AndI8().setUpdatePrecisions(false),
            {
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            },
            {
                ngraph::element::f32,
                {{}, {}, {}},
                ngraph::element::f32,
                {{ngraph::element::f32}, {1.f}, {1.f}}
            }
        },
    };
    INSTANTIATE_TEST_CASE_P(
        LPT,
        BaseTransformation,
        ::testing::ValuesIn(testValues),
        BaseTransformation::getTestCaseName);
} // namespace
