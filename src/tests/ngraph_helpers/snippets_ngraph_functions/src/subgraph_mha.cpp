// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mha.hpp"

#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> MHAFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    std::shared_ptr<ov::Node> matmul_parent1 = transpose1;
    if (with_mul) {
        std::vector<float> mulConstData(ngraph::shape_size(constantShapes[2]));
        auto mulConst = ngraph::builder::makeConstant(precision, constantShapes[2], mulConstData, true);
        matmul_parent1 = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, matmul_parent1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAFunction::initReference() const {
    auto data0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParams = {data0, data1, data2, data3};
    NodeVector subgraph_inputs = {data0, data1, data2, data3};

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    ngraph::ParameterVector subgraph_params = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    std::shared_ptr<ov::Node> matmul_parent1 = transpose1;
    if (with_mul) {
        std::vector<float> mulConstData(ngraph::shape_size(constantShapes[2]));
        auto mulConst = ngraph::builder::makeConstant(precision, constantShapes[2], mulConstData, true);
        auto mulParam = std::make_shared<ngraph::opset1::Parameter>(precision, mulConst->get_shape());
        matmul_parent1 = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulParam);
        subgraph_params = {transpose0Param, transpose1Param, mulParam, addParam, transpose2Param};
        subgraph_inputs = {data0, data1, mulConst, data2, data3};
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, matmul_parent1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_inputs,
            std::make_shared<ov::Model>(NodeVector{transpose3}, subgraph_params));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ngraphParams);
}

std::shared_ptr<ov::Model> MHAMatMul0TransposeFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    const auto order = std::vector<int64_t>{0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], order);
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], order);
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], order);
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], order);

    std::vector<float> mulConstData(1);
    auto mulConst = ngraph::builder::makeConstant(precision, ov::Shape{1}, mulConstData, true);

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul, transA, true);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAMatMul0TransposeFunction::initReference() const {
    auto data0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParams = {data0, data1, data2, data3};

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    std::vector<float> mulConstData(1);
    auto mulConst = ngraph::builder::makeConstant(precision, ov::Shape{1}, mulConstData, true);
    ngraph::ParameterVector subgraphParams = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(
            NodeVector{data0, data1, data2, data3},
            std::make_shared<ov::Model>(NodeVector{transpose3}, subgraphParams));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ngraphParams);
}

std::shared_ptr<ov::Model> MHASelectFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto less0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    auto less1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[4]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[5]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, less0Param, less1Param, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0],
                                                         std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1],
                                                         std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5],
                                                         std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6],
                                                         std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] *
                                                                   input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);
    // Value is equal to '1' - to avoid situation e^(-1000) / (sum(e^(-1000)) = 0/0 = NAN
    auto selectConst = ngraph::builder::makeConstant(precision, ov::Shape{1}, std::vector<float>{1});

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto less = std::make_shared<ngraph::opset3::Less>(less0Param, less1Param);
    std::shared_ptr<ov::Node> selectCond = less;
    if (add->get_output_partial_shape(0) != input_shapes[3]) {
        const auto broadcast_shape = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5],
                                                                   add->get_output_shape(0));
        const auto broadcast = ngraph::builder::makeBroadcast(selectCond, broadcast_shape,
                                                              ngraph::op::BroadcastType::NUMPY);
        selectCond = broadcast;
    }
    const auto select = std::make_shared<ngraph::opset1::Select>(selectCond, selectConst, add,
                                                                 ngraph::op::AutoBroadcastType::NUMPY);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(select, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    // to generate valid values
    less0Param->set_friendly_name("less0");
    less0Param->set_friendly_name("less1");

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}


std::shared_ptr<ov::Model> MHAWOTransposeFunction::initOriginal() const {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    ngraph::ParameterVector ngraphParam = {param0, param1, param2};

    std::shared_ptr<ov::Node> parent_matmul_1 = param1;
    if (m_with_mul) {
        const auto mulConst = ngraph::builder::makeConstant(precision, ov::Shape({1}), std::vector<float>{1}, true);
        parent_matmul_1 = std::make_shared<ngraph::opset3::Multiply>(param1, mulConst);
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(param0, parent_matmul_1, false, m_transposed_b_0);
    const auto softmax = std::make_shared<ngraph::opset1::Softmax>(matMul0, input_shapes[0].size() - 1);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, param2);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(matMul1)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}

std::shared_ptr<ov::Model> MHAWOTransposeOnInputsFunction::initOriginal() const {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    ngraph::ParameterVector ngraphParam = {param0, param1, param2};

    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 1, 3});

    std::shared_ptr<ov::Node> parent_matmul_1 = param1;
    if (m_with_mul) {
        const auto mulConst = ngraph::builder::makeConstant(precision, ov::Shape({1}), std::vector<float>{1}, true);
        parent_matmul_1 = std::make_shared<ngraph::opset3::Multiply>(param1, mulConst);
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(param0, parent_matmul_1, false, m_transposed_b_0);
    const auto softmax = std::make_shared<ngraph::opset1::Softmax>(matMul0, input_shapes[0].size() - 1);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, param2);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
