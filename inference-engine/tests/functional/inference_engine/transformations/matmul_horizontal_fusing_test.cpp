// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/pass/visualize_tree.hpp>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/matmul_horizontal_fusing.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

struct MatMulBuilder {
    struct WeightsPath {
        element::Type precision;
        PartialShape shape;
        std::vector<float> values;
    };

    struct BiasPath {
        Shape shape;
        std::vector<float> values;
    };

    MatMulBuilder(WeightsPath weights, BiasPath bias = {}, bool transpose_a = false, bool transpose_b = false, size_t num_splits_after = 0) :
        weights(weights), bias(bias), transpose_a(transpose_a), transpose_b(transpose_b), num_splits_after(num_splits_after) {}

    WeightsPath weights;
    BiasPath bias;
    bool transpose_a;
    bool transpose_b;
    size_t num_splits_after;
};

enum AdditionalOp {
    CLAMP,
    NONE,
};

struct MatMulHorizontalFusingTestValues {
    element::Type input_precision;
    ngraph::PartialShape input_shape;
    AdditionalOp additional_consumer;
    std::vector<MatMulBuilder> matmuls_before;
    std::vector<MatMulBuilder> matmuls_after;
};

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const AdditionalOp additional_op,
    const std::vector<MatMulBuilder>& matmul_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };
    auto relu = std::make_shared<opset8::Relu>(input);

    OutputVector concat_nodes;
    for (const auto& matmul_val : matmul_values) {
        std::shared_ptr<ngraph::Node> weights;
        if (matmul_val.weights.values.empty()) {
            auto input_2 = std::make_shared<opset8::Parameter>(matmul_val.weights.precision, matmul_val.weights.shape);
            inputs.emplace_back(input_2);
            weights = input_2;
        } else {
            auto weights_shape = matmul_val.weights.shape.to_shape();
            weights = opset8::Constant::create(matmul_val.weights.precision, weights_shape, matmul_val.weights.values);
            if (matmul_val.weights.precision != element::f32) {
                weights = std::make_shared<ngraph::opset8::Convert>(weights, element::f32);

                Shape deq_const_shape(2, 1ul);
                size_t out_channel_idx = matmul_val.transpose_b ? weights_shape.size() - 2 : weights_shape.size() - 1;
                deq_const_shape[out_channel_idx] = weights_shape[out_channel_idx];

                auto sub_const = opset8::Constant::create(element::f32, deq_const_shape, { 0.0001f });
                weights = std::make_shared<ngraph::opset8::Subtract>(weights, sub_const);

                auto mul_const = opset8::Constant::create(element::f32, deq_const_shape, { 0.56f });
                weights = std::make_shared<ngraph::opset8::Multiply>(weights, mul_const);
            }
        }

        std::shared_ptr<ngraph::Node> last_node = std::make_shared<ngraph::opset8::MatMul>(relu, weights, matmul_val.transpose_a, matmul_val.transpose_b);;

        if (!matmul_val.bias.values.empty()) {
            auto bias_const = opset8::Constant::create(element::f32, matmul_val.bias.shape, matmul_val.bias.values);
            auto bias = std::make_shared<opset8::Add>(last_node, bias_const);
            last_node = bias;
        }

        if (matmul_val.num_splits_after == 0) {
            concat_nodes.emplace_back(last_node);
        } else {
            auto split_axis = opset8::Constant::create(element::i64, Shape{}, { 2 });
            last_node = std::make_shared<opset8::Split>(last_node, split_axis, matmul_val.num_splits_after);
            auto outputs = last_node->outputs();
            for (const auto& out : outputs) {
                concat_nodes.emplace_back(out);
            }
        }
    }

    NodeVector results;
    if (concat_nodes.size() == 1) {
        results.emplace_back(concat_nodes[0].get_node_shared_ptr());
    } else {
        auto concat = std::make_shared<opset8::Concat>(concat_nodes, 0);
        results.emplace_back(concat);
    }

    if (AdditionalOp::CLAMP) {
        auto clamp = std::make_shared<opset8::Clamp>(relu, 0.0, 6.0);
        results.emplace_back(clamp);
    }

    return std::make_shared<Function>(results, inputs);
}

class MatMulHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<MatMulHorizontalFusingTestValues> {
public:
    void SetUp() override {
        const auto values = GetParam();

        f = get(values.input_precision, values.input_shape, values.additional_consumer, values.matmuls_before);

        ngraph::pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MatMulHorizontalFusion>();
        manager.run_passes(f);

        f_ref = get(values.input_precision, values.input_shape, values.additional_consumer, values.matmuls_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulHorizontalFusingTestValues> obj) {
        const auto testValues = obj.param;

        std::ostringstream result;
        result << testValues.input_shape << "_" << testValues.input_precision << "_matmuls_before_"
               << (testValues.additional_consumer == AdditionalOp::CLAMP ? "additional_op_" : "");
        for (const auto& elem : testValues.matmuls_before) {
            result << "{weights_" << elem.weights.precision << elem.weights.shape;
            if (!elem.bias.values.empty()) {
                result << "_bias_" << elem.bias.shape;
            }
            result << (elem.transpose_a ? "transpose_a_" : "") << (elem.transpose_b ? "transpose_b_" : "") << "}_";
        }
        result << "matmuls_after_";
        for (const auto& elem : testValues.matmuls_after) {
            result << "{weights_" << elem.weights.precision << elem.weights.shape;
            if (!elem.bias.values.empty()) {
                result << "_bias_" << elem.bias.shape;
            }
            result << (elem.transpose_a ? "transpose_a_" : "") << (elem.transpose_b ? "transpose_b_" : "") << "}_";
            if (elem.num_splits_after > 0) {
                result << "split_into_" << elem.num_splits_after << "_outputs";
            }
            result << "}_";
        }

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> f;
    std::shared_ptr<ngraph::Function> f_ref;
};

TEST_P(MatMulHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<MatMulHorizontalFusingTestValues> test_values {
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        // actual
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        // expected
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::CLAMP,
        // actual
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        // expected
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, false, true},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, true, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 2 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, true, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, true, true},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, true, true, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 8 }, {2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4}},
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape::dynamic(3), AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} },
                MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, { 15.f} }
            },
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} },
                MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, { 30.f} }
            },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{Shape{ 1, 1, 8 }, { 15.f, 15.f, 15.f, 15.f, 30.f, 30.f, 30.f, 30.f }},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} },
                MatMulBuilder::BiasPath{ Shape{ 4 }, { 15.f} }
            },
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} },
                MatMulBuilder::BiasPath{ Shape{ 4 }, { 30.f} }
            },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{Shape{ 8 }, { 15.f, 15.f, 15.f, 15.f, 30.f, 30.f, 30.f, 30.f }},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {6} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 12 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f}},
                MatMulBuilder::BiasPath{}, false, false, 3
            }
        }
    },
    // fused only 2 matmuls
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6} }, MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, {15.f}}},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, false, 2
            },
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6} }, MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, {15.f}}}
        }
    },
    // not transformed: dynamic rank
    {
        element::f32, PartialShape::dynamic(), AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        }
    },
    // not transformed: different weights precision
    {
        element::f32, PartialShape::dynamic(), AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        }
    },
    // not transformed: different transpose flags
    {
        element::f32, PartialShape{ 1, 4, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, true, false},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, true, false},
        }
    },
    // not transformed: matmul with two activations
    {
        element::f32, PartialShape{ 1, 4, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {} }},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    MatMulHorizontalFusing,
    ::testing::ValuesIn(test_values),
    MatMulHorizontalFusing::getTestCaseName);
} // namespace
