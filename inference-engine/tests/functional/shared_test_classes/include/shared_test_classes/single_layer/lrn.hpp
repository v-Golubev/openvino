// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using lrnLayerTestParamsSet = std::tuple<
        double,                        // Alpha
        double,                        // Beta
        double,                        // Bias
        size_t,                        // Size
        std::vector<int64_t>,          // Reduction axes
        ov::test::ElementType,         // Network precision
        ov::test::ElementType,         // Input precision
        ov::test::ElementType,         // Output precision
        ov::test::InputShape,          // Input shapes
        std::string                    // Device name
>;

class LrnLayerTest
        : public testing::WithParamInterface<lrnLayerTestParamsSet>,
          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<lrnLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
