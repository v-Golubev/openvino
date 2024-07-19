// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
/**
 * @interface BrgemmTPPBlocking
 * @brief Covers BrgemmTPP with blocking loops
 * @ingroup snippets
 */

class BrgemmTPPBlocking : public ov::snippets::lowered::pass::BrgemmBlockingBase {
public:
    OPENVINO_RTTI("BrgemmTPPBlocking", "BrgemmBlockingBase")

private:
    bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) override;
};

}  // namespace pass
}  // namespace tpp
}  // namespace intel_cpu
}  // namespace ov