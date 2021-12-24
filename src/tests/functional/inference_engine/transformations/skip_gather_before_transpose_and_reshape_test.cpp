// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp>

#include <ngraph/function.hpp>
#include <openvino/opsets/opset7.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/utils/utils.hpp>

#include <transformations/serialize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeStaticShapeFpData) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto transpose_const = opset7::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset7::Transpose>(data, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeStaticShapeIntData) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset7::Parameter>(element::i64, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::i64, data_shape);

        auto transpose_const = opset7::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset7::Transpose>(data, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeDynamicShapeStaticBatch) {
    PartialShape data_shape{1, -1, -1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto transpose_const = opset7::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset7::Transpose>(data, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeIncorrectGatherAxis) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {2});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeDynamicBatch) {
    PartialShape data_shape{-1, -1, -1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeBatchNotEqualTo1) {
    PartialShape data_shape{2, 3, 12, 12};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto indices_node = opset7::Constant::create(element::i64, {}, {0});
        auto axis_node = opset7::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset7::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset7::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset7::Transpose>(gather, transpose_const);

        auto reshape_const = opset7::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_const, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}