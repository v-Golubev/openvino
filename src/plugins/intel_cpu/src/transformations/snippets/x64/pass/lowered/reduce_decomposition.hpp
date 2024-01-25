// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface ReduceDecomposition
 * @brief Decomposes snippets::Reduce operations to a range of low-level operations on linear IR
 * @attention Only Reduce by last dimension is supported
 * @ingroup snippets
 */
class ReduceDecomposition : public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("ReduceDecomposition", "Pass")
    explicit ReduceDecomposition(size_t vector_size);
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_vector_size;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
