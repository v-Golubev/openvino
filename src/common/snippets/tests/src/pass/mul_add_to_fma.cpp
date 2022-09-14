// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pass/mul_add_to_fma.hpp>
#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include <subgraph_converts.hpp>
#include "snippets/pass/mul_add_to_fma.hpp"

namespace ov {
namespace test {
namespace snippets {

void MulAddToFMATests::run() {
    ASSERT_TRUE(function);
    std::string name;
    manager.register_pass<ngraph::snippets::pass::MulAddToFMA>();
}

TEST_F(MulAddToFMATests, smoke_Snippets_MulAddToFMAMulAs0AddInput) {
    const size_t in_idx_for_add = 1ul;
    const auto &f = EltwiseWithMulAddFunction(std::vector<Shape> {{1, 3, 2, 2}, {1, 3, 2, 2}, {1, 3, 2, 2}}, in_idx_for_add);
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(MulAddToFMATests, smoke_Snippets_MulAddToFMAMulAs1AddInput) {
    const size_t in_idx_for_add = 1ul;
    const auto &f = EltwiseWithMulAddFunction(std::vector<Shape> {{1, 3, 2, 2}, {1, 3, 2, 2}, {1, 3, 2, 2}}, in_idx_for_add);
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(MulAddToFMATests, smoke_Snippets_MulAddToFMANegative) {
    auto get_f = [&]() {
        auto data0 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        auto data2 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});

        auto mul = std::make_shared<op::v1::Multiply>(data0, data1);
        auto additional_consumer = std::make_shared<op::v0::Relu>(mul);
        auto add = std::make_shared<op::v1::Add>(mul, data2);

        return std::make_shared<Model>(ov::NodeVector{add, additional_consumer}, ov::ParameterVector{data0, data1, data2});
    };

    function = get_f();
    function_ref = get_f();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov