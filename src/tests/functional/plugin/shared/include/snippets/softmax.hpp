// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                      // Input 0 Shape
        int,                             // Axis
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> SoftmaxParams;

typedef std::tuple<
        std::pair<InputShape, InputShape>,// Input Shapes
        int,                              // Axis
        size_t,                           // Expected num nodes
        size_t,                           // Expected num subgraphs
        std::string                       // Target Device
> AddSoftmaxParams;

class Softmax : public testing::WithParamInterface<ov::test::snippets::SoftmaxParams>,
                virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::SoftmaxParams> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& dataShape = targetInputStaticShapes[0];
        const auto funcInput = function->inputs()[0];
        auto tensor = ov::Tensor{funcInput.get_element_type(), {dataShape}};
        auto begin = tensor.data<float>();
        std::vector<float> vals(ov::shape_size(dataShape), std::numeric_limits<float>::infinity());
        std::copy(vals.begin(), vals.end(), begin);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
};

class AddSoftmax : public testing::WithParamInterface<ov::test::snippets::AddSoftmaxParams>,
                   virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddSoftmaxParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov