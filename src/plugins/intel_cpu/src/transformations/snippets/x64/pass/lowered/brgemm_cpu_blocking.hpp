// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmCPUBlocking
 * @brief Covers BrgemmCPU with blocking loops
 * @ingroup snippets
 */
class BrgemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlockingBase {
public:
    OPENVINO_RTTI("BrgemmCPUBlocking", "BrgemmBlockingBase")

private:
    bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) override;

    static snippets::lowered::LinearIR::constExprIt move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir,
                                                                           const snippets::lowered::LinearIR::constExprIt& brgemm_it);

    static snippets::lowered::LinearIR::constExprIt get_loop_begin_pos(snippets::lowered::LinearIR& linear_ir,
                                                                       const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                                                                       const snippets::lowered::ExpressionPtr& copy_b_expr);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov