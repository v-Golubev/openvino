// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface BrgemmBlockingBase
 * @brief Base class for Brgemm blocking loops markup
 * @ingroup snippets
 */
class BrgemmBlockingBase : public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("BrgemmBlockingBase", "RangedPass")
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

protected:
    /**
     * @interface mark_blocking_loops
     * @brief Covers brgemm with blocking loops. Also should calculate optimal blocking parameters inside.
     * @param linear_ir LIR that contain's brgemm
     * @param brgemm_it iterator on brgemm expression which should be covered with blocking loops
     */
    virtual bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) = 0;
    // virtual std::tuple<size_t, size_t, size_t> get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) = 0;

    static bool blocking_loop_exists(const snippets::lowered::LoopManagerPtr& loop_manager,
                                     const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                                     const std::shared_ptr<ov::snippets::op::Brgemm>& brgemm);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov