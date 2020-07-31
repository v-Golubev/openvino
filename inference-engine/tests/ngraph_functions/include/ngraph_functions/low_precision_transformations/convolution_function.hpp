// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precisionBeforeDequantization,
        ngraph::builder::subgraph::DequantizationOperations dequantization,
        ngraph::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precisionBeforeDequantization,
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        ngraph::element::Type precisionAfterOperation,
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
        ngraph::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
