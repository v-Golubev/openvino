// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_external_parameters.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/rt_info/external_parameter.hpp"

namespace ov::intel_cpu::pass {
bool RemoveExternalParameters::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBuffers")

    const auto& params = linear_ir.get_parameters();
    bool modified = false;
    std::vector<snippets::lowered::LinearIR::constExprIt> params_to_remove;
    for (const auto& param : params) {
        if (ov::snippets::is_external_parameter(param->get_node())) {
            const auto param_it = linear_ir.find(param);
            std::cout << " [ INFO ] Parameter with name " << param->get_node()->get_friendly_name() << " has been removed\n";
            params_to_remove.push_back(param_it);
            modified = true;
        }
    }
    for (const auto& param_it : params_to_remove)
        linear_ir.erase(param_it);
    return modified;
}
}  // namespace ov::intel_cpu::pass
