// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DequantizationOperations {
public:
    class Convert {
    public:
        Convert();
        Convert(const ngraph::element::Type outPrecision);
        bool empty() const noexcept;

        ngraph::element::Type outPrecision;
    private:
        bool isEmpty;
    };

    class Subtract {
    public:
        Subtract();
        Subtract(const float value);
        Subtract(const std::vector<float>& values);
        Subtract(const std::vector<float>& values, const ngraph::element::Type outPrecision);
        Subtract(const std::vector<float>& values, const ngraph::element::Type outPrecision, const ngraph::Shape& constantShape);
        bool empty() const noexcept;

        std::vector<float> values;
        ngraph::element::Type outPrecision;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined;
    private:
        bool isEmpty;
    };

    class Multiply {
    public:
        Multiply();
        Multiply(const float value);
        Multiply(const std::vector<float>& values);
        Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision);
        Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision, const ngraph::Shape& constantShape);
        bool empty() const noexcept;

        std::vector<float> values;
        ngraph::element::Type outPrecision;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined;
    private:
        bool isEmpty;
    };

    DequantizationOperations();

    DequantizationOperations(const Convert& convert, const Subtract& subtract, const Multiply& multiply);

    bool empty() const;

    Convert convert;
    Subtract subtract;
    Multiply multiply;
};

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" <<
        data.convert.outPrecision << "_" <<
        data.subtract.values << "_" <<
        data.multiply.values;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
