// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {

class SplitFunction {
public:
    class ActualValues {
    public:
        ngraph::element::Type lowPrecision;
        ngraph::Shape subtractShape;
        std::vector<float> subtractValues;
        ngraph::Shape multiplyShape;
        std::vector<float> multiplyValues;
        int64_t splitedAxis;
        size_t numSplit;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type lowPrecision;
        ngraph::Shape subtractShape;
        std::vector<std::vector<float>> subtractValues;
        ngraph::Shape multiplyShape;
        std::vector<std::vector<float>> multiplyValues;
        int64_t splitedAxis;
        size_t numSplit;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const int64_t splitedAxis,
        const size_t numSplit);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ExpectedValues& values);
};

inline std::ostream& operator<<(std::ostream& out, const SplitFunction::ActualValues& values) {
    std::ostringstream result;
    result << "_" << values.lowPrecision <<
        "subtract_shape=" << values.subtractShape << "_subtract_values=" << values.subtractValues <<
        "_multiply_shape=" << values.multiplyShape << "_multiplly_values=" << values.multiplyValues <<
        "_axis=" << values.splitedAxis << "_n_slpits=" << values.numSplit;

    return out << result.str();
}

inline std::ostream& operator<<(std::ostream& out, const SplitFunction::ExpectedValues& values) {
    std::ostringstream result;
    result << "_" << values.lowPrecision <<
        "subtract_shape=" << values.subtractShape << "_subtract_values={ ";
    for (auto& val : values.subtractValues) {
        result << val << ", ";
    }
    result << "}_multiply_shape=" << values.multiplyShape << "_multiply_values = {";
    for (auto& val : values.multiplyValues) {
        result << val << ", ";
    }
    result << "}";
    return out << result.str();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
