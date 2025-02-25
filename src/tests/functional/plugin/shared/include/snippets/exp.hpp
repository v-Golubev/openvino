// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> ExpParams;

class Exp : public testing::WithParamInterface<ov::test::snippets::ExpParams>,
            virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::ExpParams> obj);

protected:
    void SetUp() override;
};

class SubExp : public Exp {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

class ExpReciprocal : public Exp {
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov