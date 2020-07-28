// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

DequantizationOperations::Convert::Convert() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined)
{}

DequantizationOperations::Convert::Convert(const ngraph::element::Type outPrecision) :
    isEmpty(false),
    outPrecision(outPrecision)
{}

bool DequantizationOperations::Convert::empty() const noexcept {
    return isEmpty;
}

DequantizationOperations::Subtract::Subtract() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false)
{}

DequantizationOperations::Subtract::Subtract(const float value) :
    isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(const std::vector<float>& values, const ngraph::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& constantShape) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantShapeIsDefined(true) {
}

bool DequantizationOperations::Subtract::empty() const noexcept {
    return isEmpty;
}

DequantizationOperations::Multiply::Multiply() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const float value) :
    isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& constantShape) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantShapeIsDefined(true) {
}

bool DequantizationOperations::Multiply::empty() const noexcept {
    return isEmpty;
}

DequantizationOperations::DequantizationOperations() {}

DequantizationOperations::DequantizationOperations(
    const Convert& convert,
    const Subtract& subtract,
    const Multiply& multiply) :
    convert(convert),
    subtract(subtract),
    multiply(multiply)
{}

bool DequantizationOperations::empty() const {
    return convert.empty() && subtract.empty() && multiply.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
