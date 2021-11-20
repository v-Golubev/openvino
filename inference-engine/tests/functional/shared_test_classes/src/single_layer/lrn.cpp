// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/lrn.hpp"

namespace LayerTestsDefinitions {

std::string LrnLayerTest::getTestCaseName(const testing::TestParamInfo<lrnLayerTestParamsSet>& obj) {
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    ov::test::ElementType netPrecision;
    ov::test::ElementType inType, outType;
    ov::test::InputShape inputShapes;
    std::string targetDevice;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inType, outType, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS=" << CommonTestUtils::partialShape2str({ inputShapes.first }) << separator << "TS=(";
    for (const auto& shape : inputShapes.second) {
        result << CommonTestUtils::vec2str(shape) << separator;
    }
    result << ")" << separator;
    result << "Alpha=" << alpha << separator;
    result << "Beta=" << beta << separator;
    result << "Bias=" << bias << separator;
    result << "Size=" << size << separator;
    result << "Axes=" << CommonTestUtils::vec2str(axes) << separator;
    result << "netPRC=" << netPrecision << separator;
    result << "inPRC=" << inType << separator;
    result << "outPRC=" << outType << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void LrnLayerTest::SetUp() {
    ov::test::InputShape inputShapes;
    ov::test::ElementType netPrecision;
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inType, outType, inputShapes, targetDevice) = GetParam();

    init_input_shapes({ inputShapes });

    auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
    auto paramIn = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto axes_node = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{axes.size()}, axes.data());
    auto lrn = std::make_shared<ngraph::opset3::LRN>(paramIn[0], axes_node, alpha, beta, bias, size);
    ngraph::ResultVector results {std::make_shared<ngraph::opset3::Result>(lrn)};
    function = std::make_shared<ngraph::Function>(results, params, "lrn");
}
}  // namespace LayerTestsDefinitions
