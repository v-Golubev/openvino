// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface RemoveExternalParameters
 * @brief TODO
 * @ingroup snippets
 */
class RemoveExternalParameters : public snippets::lowered::pass::Pass {
public:
    RemoveExternalParameters() = default;
    OPENVINO_RTTI("RemoveExternalParameters", "", Pass)
    bool run(snippets::lowered::LinearIR& linear_ir) override;
};

}  // namespace ov::intel_cpu::pass
