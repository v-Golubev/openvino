// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AssignRegisters
 * @brief Assigns in/out abstract registers indexes to every operation.
 * Note that changing of the IR is likely to invalidate register assignment.
 * @ingroup snippets
 */
class AssignRegisters : public ConstRangedPass {
public:
    OPENVINO_RTTI("AssignRegisters", "Pass")
    explicit AssignRegisters(const std::function<Generator::opRegType(const std::shared_ptr<Node>& op)>& mapper) : m_reg_type_mapper(mapper) {}
    bool run(const LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    std::function<Generator::opRegType(const std::shared_ptr<Node>& op)> m_reg_type_mapper;
    static constexpr size_t reg_count = 16lu;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
