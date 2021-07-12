// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "cpu_types.h"
#include "mkldnn_eltwise_node.h"

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        const auto shapeA = matMul->get_input_shape(0);
        const auto shapeB = matMul->get_input_shape(1);

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_shape(i).size();
            if (inShapeRank < 2 || inShapeRank > 3) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_shape().size();
        if (outShapeRank < 2 || outShapeRank > 3) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatMulNode::MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "MatMul node with name '" + getName() + "'";

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MKLDNNMatMulNode::canFuse(const MKLDNNNodePtr& node) const {
    return one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                  EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                  EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu);
}

void MKLDNNMatMulNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) const {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}


std::shared_ptr<mkldnn::primitive_attr> MKLDNNMatMulNode::initPrimitiveAttr() const {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, true);

    return attr;
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);

    auto minSize = [](const Precision lhs, const Precision rhs) {
        return lhs.size() < rhs.size();
    };

    if (firstInPortPrec != secondInPortPrec)
        firstInPortPrec = secondInPortPrec = std::min(firstInPortPrec, secondInPortPrec, minSize);

    auto firstInDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(firstInPortPrec);
    auto secondInDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(secondInPortPrec);
    auto outputDataType = firstInDataType;

    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }

    inDims.reserve(2);
    inDims[0] = getParentEdgeAt(0)->getDims();
    inDims[1] = getParentEdgeAt(1)->getDims();
    outDims[0] = getChildEdgeAt(0)->getDims();

    if (inDims[0].ndims() != inDims[1].ndims() || inDims[0].ndims() != outDims[0].ndims())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    int nDims = inDims[0].ndims();
    auto xAxis = nDims - 1;
    auto yAxis = nDims - 2;
    auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    // coverity[copy_paste_error]
    if (inDims[0][xAxis0] != inDims[1][yAxis1] ||
        inDims[0][yAxis0] != outDims[0][yAxis] ||
        inDims[1][xAxis1] != outDims[0][xAxis])
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((inDims[0][dim_idx] != outDims[0][dim_idx] && inDims[0][dim_idx] != 1) ||
            (inDims[1][dim_idx] != outDims[0][dim_idx] && inDims[1][dim_idx] != 1)) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    /* Example MatMul:
     * 2x128x512(T) * 2x128x512 = 2x512x512
     * First input 2x128x512(T) should be transposed
     * oneDNN requires memory::desc for this input to:
     * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
     * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
     */
    auto getStridesAndDims = [](MKLDNNDims& dims, const bool transpose) {
        const auto ndims = dims.ndims();

        mkldnn::memory::dims strides(ndims, 1);
        for (size_t i = 1; i < ndims; i++) {
            strides[ndims - i - 1 ] = strides[ndims - i] * dims[ndims - i];
        }

        if (transpose && ndims > 1) {
            std::swap(dims[ndims - 2], dims[ndims - 1]);
            strides[ndims - 1] = dims[ndims - 2];
            strides[ndims - 2] = 1;
        }

        return strides;
    };

    mkldnn::memory::dims inStrides0 = getStridesAndDims(inDims[0], transposeIn[0]);
    mkldnn::memory::dims inStrides1 = getStridesAndDims(inDims[1], transposeIn[1]);
    mkldnn::memory::dims outStrides = getStridesAndDims(outDims[0], false);

    in_data_d[0] = {inDims[0], firstInDataType, inStrides0};
    in_data_d[1] = {inDims[1], secondInDataType, inStrides1};
    out_data_d = {outDims[0], outputDataType, outStrides};

    createDescriptor({in_data_d[0], in_data_d[1]}, {out_data_d});
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                        const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNDescriptor desc{
        std::shared_ptr<matmul::desc>(
            new matmul::desc(in_data_d[0], in_data_d[1], out_data_d))};

    descs.push_back(desc);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);
        while (static_cast<bool>(itpd)) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < 2; i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getSrcMemDesc(itpd, i));
                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getDstMemDesc(itpd, i));
                config.outConfs.push_back(dataConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNMatMulNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    if (prim)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
    std::shared_ptr<matmul::primitive_desc> prim_desc;
    prim_desc = std::make_shared<matmul::primitive_desc>(
            createPrimitiveDescriptor<matmul::primitive_desc, matmul::desc>(*attr));

    prim.reset(new matmul(*prim_desc));

    auto src0 = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto src1 = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    primArgs = {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_WEIGHTS_0, src1}, {DNNL_ARG_DST, dst}};
}

MKLDNNMemoryDesc MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
        : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

    MKLDNNDims parentDims;
    parentDims = getParentEdgeAt(idx)->getDims();

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            parentDims.ToSizeVector(),
                                                            desc.getLayout()));
    } else if (transposeIn[idx]) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            TensorDesc::getLayoutByDims(getParentEdgeAt(idx)->getDims().ToSizeVector())));
    } else {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            parentDims.ToSizeVector(),
                                                            desc.getBlockingDesc()));
    }
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

int MKLDNNMatMulNode::getMaxBatch() {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
