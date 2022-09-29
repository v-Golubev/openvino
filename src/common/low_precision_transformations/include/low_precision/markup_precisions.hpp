// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <vector>

#include <ngraph/pass/pass.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/common/precisions_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupPrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Transformation is used to add customization options runtime
/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupPrecisions transformation marks:
 *    1) not supported operations by PrecisionsAttribute attribute with empty precisions,
 *    2) operations with required precisions by PrecisionsAttribute attribute according to the provided restrictions,
 *    3) precision preserved operations by PrecisionPreservedAttribute attribute.
 *
 * For more details about the transformation, refer to
 * [MarkupPrecisions](@ref openvino_docs_OV_UG_lpt_MarkupPrecisions) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::MarkupPrecisions : public ngraph::pass::FunctionPass {
public:
    class Restriction {
    public:
        explicit Restriction(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const uint64_t version, const ngraph::pass::low_precision::PrecisionsRestriction::PrecisionsByPorts& precisions) {
            precisionsByVersion.emplace(version, precisions);
        }

        bool versionIsRequired;
        std::unordered_map<uint64_t, ngraph::pass::low_precision::PrecisionsRestriction::PrecisionsByPorts> precisionsByVersion;
    };

    OPENVINO_RTTI("MarkupPrecisions", "0");
    explicit MarkupPrecisions(const std::vector<PrecisionsRestriction>& restrictions = {},
        const std::vector<ngraph::element::Type>& defaultPrecisions = { ngraph::element::u8, ngraph::element::i8 });
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    static bool isPrecisionPreserved(const std::shared_ptr<Node>& node);
    static bool isSupported(const std::shared_ptr<Node>& node);
    std::unordered_map<std::string, Restriction> restrictionsByOperation;
    std::vector<ngraph::element::Type> defaultPrecisions;
};
