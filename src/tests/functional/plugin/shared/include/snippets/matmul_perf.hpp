// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<ov::PartialShape>, // Input  Shapes
        std::vector<ov::element::Type>,// Input Element types
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        bool,                          // Enable/Disable tokenization
        std::string                    // Target Device
> MatMulPerfParams;

typedef std::tuple<
        std::vector<ov::PartialShape>,   // Input shapes
        bool,                            // With Multiply
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        bool,                          // Enable/Disable tokenization
        std::string                      // Target Device
> MHAParamsPerf;

class MatMulPerf : public testing::WithParamInterface<ov::test::snippets::MatMulPerfParams>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulPerfParams> obj);

protected:
    void SetUp() override;
};

class MHAWOTransposePerf : public testing::WithParamInterface<ov::test::snippets::MHAParamsPerf>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParamsPerf> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
};

} // namespace snippets
} // namespace test
} // namespace ov