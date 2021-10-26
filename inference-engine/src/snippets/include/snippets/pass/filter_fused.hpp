// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface FilterFused
 * @brief Mark operations that will be fused on plugin side (but not yet in snippets) so they'll be ignored by snippets.
 * @ingroup snippets
 */
class TRANSFORMATIONS_API FilterFused : public ngraph::pass::FunctionPass {
public:
    FilterFused() : FunctionPass() {
        //set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};
/*
 FusedWithConvolution, FusedWithConvolutionSumActivation, FusedWithMisc - fusing chain is active and may be continued
 FusedTerminator - the node is fused, but the chain must be interrupted
 Ignored - must be skipped, since can't be handled properly at this time
 */
enum class SnippetsNodeType : int64_t {NotSet = 0, FusedWithConvolution, FusedWithConvolutionSumActivation,
                                        FusedWithMisc, FusedTerminator, Ignored, SubgraphStart, SubgraphBody};
void SetSnippetsNodeType(std::shared_ptr<Node> node, SnippetsNodeType);
SnippetsNodeType GetSnippetsNodeType(std::shared_ptr<Node> node);

} // namespace pass
} // namespace snippets
} // namespace ngraph
