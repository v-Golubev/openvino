// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface OptimizeLoopSingleEvaluation
 * @brief Does the following optimizations if the Loop body can be executed only once:
 *        - sets evaluate_once parameter to true
 *        - moves all ptr arithmetic to finalization offsets
 * @ingroup snippets
 */
class OptimizeLoopSingleEvaluation : public IsolatedRangedPass {
public:
    OPENVINO_RTTI("OptimizeLoopSingleEvaluation", "Pass")
    bool run(lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
