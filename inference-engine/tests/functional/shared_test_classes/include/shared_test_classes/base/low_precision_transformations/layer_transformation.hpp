// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "low_precision/iparams_manager.hpp"
#include "low_precision/ilayer_transformations_manager.hpp"
#include "low_precision/layer_transformation.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsUtils {

class LayerTransformationParamsNGraphFactory {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParams();
};

class LayerTransformationParamsFactory : public LayerTransformationParamsNGraphFactory {
};

IE_SUPPRESS_DEPRECATED_START

class LayerTransformation : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    // TODO: LPT: not implemented: clean up ngraph::pass::low_precision::LayerTransformation::Params, use this type instead
//    class Params : public ngraph::pass::low_precision::LayerTransformation::Params {
//    public:
//        Params(
//            const bool updatePrecisions = true,
//            const ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations =
//                ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
//            const ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights =
//                ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::None,
//            bool supportAsymmetricQuantization = true,
//            std::vector<ngraph::element::Type> precisionsOnActivations = { ngraph::element::u8, ngraph::element::i8 },
//            std::vector<ngraph::element::Type> precisionsOnWeights = { ngraph::element::i8 },
//            ngraph::element::Type deqPrecision = ngraph::element::f32,
//            bool support3DTensorOnActivations = true,
//            bool deconvolutionSpecificChannelsRatio = false) : ngraph::pass::low_precision::LayerTransformation::Params(
//                updatePrecisions,
//                quantizedTensorAlignmentOnActivations,
//                quantizedTensorAlignmentOnWeights,
//                supportAsymmetricQuantization,
//                deqPrecision,
//                support3DTensorOnActivations,
//                deconvolutionSpecificChannelsRatio) {}
//    };

protected:
    LayerTransformation();

    static InferenceEngine::Blob::Ptr GenerateInput(
        const ngraph::element::Type precision,
        const InferenceEngine::TensorDesc& tensorDesc,
        const float k = 1.f);

    static std::pair<float, float> getQuantizationInterval(const ngraph::element::Type precision);

    static std::string toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static InferenceEngine::Precision getDeviceInternalPrecision(const InferenceEngine::Precision precision);

    static std::string getTestCaseNameByParams(
        const InferenceEngine::Precision precision,
        const InferenceEngine::SizeVector& inputShapes,
        const std::string& targetDevice,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShapes,
        const std::string& targetDevice,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);
};

IE_SUPPRESS_DEPRECATED_END

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
