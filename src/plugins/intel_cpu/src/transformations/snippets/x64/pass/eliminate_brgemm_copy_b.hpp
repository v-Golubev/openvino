// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/external_repacking_config.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface EliminateBrgemmCopyB
 * @brief EliminateBrgemmCopyB identifies BrgemmCopyB nodes which can be inferred outside the Subgraph.
 * If this is possible, CopyB node is removed, and the external repacking is configured on the further pipeline stages in RuntimeConfigurator.
 * 
 * @ingroup snippets
 */
class EliminateBrgemmCopyB: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("EliminateBrgemmCopyB", "0");
    EliminateBrgemmCopyB(const ov::intel_cpu::ExternalRepackingConfigPtr& external_repacking_config)
        : m_external_repacking_config(external_repacking_config) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_cpu::ExternalRepackingConfigPtr m_external_repacking_config = nullptr;
};


}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
