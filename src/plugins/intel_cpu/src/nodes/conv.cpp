// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv.h"
#include "reorder.h"
#include "input.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "pooling.h"
#include "concat.h"
#include <graph.h>
#include "cpu/x64/cpu_isa_traits.hpp"
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <extension_utils.h>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "common/cpu_convert.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include <common/primitive_hashing_utils.hpp>

using namespace mkldnn;
using namespace ov::intel_cpu;
using namespace InferenceEngine;

namespace {

struct ConvKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;

    std::vector<size_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;

    mkldnn::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const ConvKey& rhs) const;
};

size_t ConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(ptr->getDnnlDesc().data));
        }
    }

    seed = get_vector_hash(seed, stride);
    seed = get_vector_hash(seed, dilation);
    seed = get_vector_hash(seed, paddingL);
    seed = get_vector_hash(seed, paddingR);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool ConvKey::operator==(const ConvKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }

    retVal = retVal && stride == rhs.stride;
    retVal = retVal && dilation == rhs.dilation;
    retVal = retVal && paddingL == rhs.paddingL;
    retVal = retVal && paddingR == rhs.paddingR;

    retVal = retVal && *attr.get() == *rhs.attr.get() && implType == rhs.implType;
    return retVal;
}

} // namespace

class MKLDNNConvolutionNode::FusedSubgraph {
public:
    FusedSubgraph(const std::vector<MKLDNNNodePtr> &opList, const MKLDNNConvolutionNode &conv, MKLDNNWeightsSharing::Ptr weightCache) {
        _graph = std::unique_ptr<MKLDNNGraph>(new MKLDNNGraph());

        std::unordered_set<MKLDNNNodePtr> nodesSet;
        std::vector<MKLDNNEdgePtr> edges;

        auto addEdge = [&](const MKLDNNNodePtr& parent, const MKLDNNNodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<MKLDNNEdge>(parent, child, parentPort, childPort);
            child->addEdge(edge);
            edges.push_back(edge);
            nodesSet.insert(parent);
            nodesSet.insert(child);
        };

        //Make inputs
        const auto &inpMemDesc1 = conv.getBaseMemDescAtOutputPort(0);
        auto inp0 = std::make_shared<MKLDNNInputNode>(inpMemDesc1, "inp0", "Parameter", conv.getEngine(), weightCache);
        inputs.push_back(inp0);
        const size_t sumPortNum = conv.getParentEdges().size() - 1;
        const auto &inpMemDesc2 = conv.getBaseMemDescAtInputPort(sumPortNum);
        auto inp1 = std::make_shared<MKLDNNInputNode>(inpMemDesc2, "inp1", "Parameter", conv.getEngine(), weightCache);
        inputs.push_back(inp1);

        auto itr = std::find_if(opList.begin(), opList.end(), [](const MKLDNNNodePtr &node) {
            if (auto eltwise = std::dynamic_pointer_cast<MKLDNNEltwiseNode>(node)) {
                return eltwise->isSpecialConvolutionAddFusing();
            }
            return false;
        });

        if (itr == opList.end())
            return;

        auto sumNode = *itr;
        addEdge(inp0, sumNode, 0, 0);
        addEdge(inp1, sumNode, 0, 1);

        //Replicate the rest of the subgraph
        auto parentItr = itr;
        while (++itr != opList.end()) {
            auto parentNode = *parentItr;
            auto currentNode = *itr;
            if (FakeQuantize == currentNode->getType()) {
                parentNode->addFusedNode(currentNode);
            } else {
                addEdge(parentNode, currentNode, 0, 0);
                auto constantsItr = conv.fusedConstNodes.find(currentNode);
                if (constantsItr != conv.fusedConstNodes.end()) {
                    size_t inpPort = 1lu;
                    for (const auto& item : constantsItr->second) {
                        addEdge(item, currentNode, 0, inpPort++);
                    }
                }
                parentItr = itr;
            }
        }

        //Make output
        const auto &outMemDesc = conv.getBaseMemDescAtOutputPort(0);
        auto out = std::make_shared<MKLDNNInputNode>(outMemDesc, "out", "Result", conv.getEngine(), weightCache);
        addEdge(*parentItr, out, 0, 0);
        outputs.push_back(out);

        std::vector<MKLDNNNodePtr> nodes(nodesSet.begin(), nodesSet.end());

        _graph->CreateGraph(nodes, edges, weightCache, "fused_subgraph");
    }

    std::shared_ptr<MKLDNNInputNode> getInput(size_t idx) const {
        if (idx < inputs.size()) {
            return inputs[idx];
        } else {
            IE_THROW(OutOfBounds) << "Unexpected input index in MKLDNNConvolutionNode::fusedSubgraph::getInput idx=" << idx
                                  << " inputs.size()=" << inputs.size();
        }
    }

    std::shared_ptr<MKLDNNInputNode> getOutput(size_t idx) const {
        if (idx < outputs.size()) {
            return outputs[idx];
        } else {
            IE_THROW(OutOfBounds) << "Unexpected output index in MKLDNNConvolutionNode::fusedSubgraph::getInput idx=" << idx
                                  << " inputs.size()=" << outputs.size();
        }
    }

    void infer() {
        _graph->ResetInferCount();
        _graph->Infer();
    }

private:
    std::unique_ptr<MKLDNNGraph> _graph;
    std::vector<std::shared_ptr<MKLDNNInputNode>> inputs;
    std::vector<std::shared_ptr<MKLDNNInputNode>> outputs;
};

bool MKLDNNConvolutionNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ngraph::is_type<ngraph::op::v1::Convolution>(op) && !ngraph::is_type<ngraph::op::v1::GroupConvolution>(op)) {
            errorMessage = "Only opset1 Convolution and GroupConvolution operations are supported";
            return false;
        }
        size_t ndims = op->get_input_partial_shape(0).rank().get_length();
        if ((ndims < 3) || (ndims > 5)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(ndims);
            return false;
        }
        if (op->get_input_partial_shape(1).is_dynamic()) {
            errorMessage = "Doesn't support dynamic weights shape";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNConvolutionNode::MKLDNNConvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), withBiases(false), withSum(false), withDWConv(false),
          isGrouped(false), dw_conv_oc(0), dw_conv_ih(0), dw_conv_iw(0), dw_conv_in_dt(memory::data_type::undef),
          groupNum(1lu), IC(1), groupIC(1), groupOC(1), eltwisePrecision(Precision::FP32) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto convolutionOp = ngraph::as_type_ptr<ngraph::op::v1::Convolution>(op);
    auto groupConvolutionOp = ngraph::as_type_ptr<ngraph::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = ConvolutionCommon;

        groupNum = 1;
        isGrouped = false;

        weightDims = convolutionOp->input_value(1).get_shape();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        biasesDims = { groupOC };

        for (int i = 0; i < convolutionOp->get_strides().size(); i++) {
            stride.push_back(convolutionOp->get_strides()[i]);
        }
        for (int i = 0; i < convolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(convolutionOp->get_dilations()[i]) - 1);
        }
        paddingL = convolutionOp->get_pads_begin();
        paddingR = convolutionOp->get_pads_end();
        autoPadding = one_of(convolutionOp->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);
    } else if (groupConvolutionOp) {
        algorithm = ConvolutionGrouped;

        groupNum = groupConvolutionOp->input_value(1).get_shape()[0];
        isGrouped = true;

        weightDims = groupConvolutionOp->input_value(1).get_shape();

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        biasesDims = {groupOC * groupNum};

        for (int i = 0; i < groupConvolutionOp->get_strides().size(); i++) {
            stride.push_back(groupConvolutionOp->get_strides()[i]);
        }
        for (int i = 0; i < groupConvolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(groupConvolutionOp->get_dilations()[i]) - 1);
        }
        paddingL = groupConvolutionOp->get_pads_begin();
        paddingR = groupConvolutionOp->get_pads_end();
        autoPadding = one_of(groupConvolutionOp->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);
    }
}

bool MKLDNNConvolutionNode::canBeExecutedInInt8() const {
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!inputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(1));
    if (!weightsZeroPoints.empty())
        weightsDataType = memory::data_type::s8;

    return one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) && weightsDataType == memory::data_type::s8;
}

InferenceEngine::Precision MKLDNNConvolutionNode::fusedEltwisePrecision(const MKLDNNNodePtr& fusingNode) const {
    InferenceEngine::Precision eltwisePrecision;

    int fusingPort = fusingNode->getFusingPort();
    if (fusingPort == 0) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(1);
    } else if (fusingPort == 1) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(0);
    } else {
        IE_THROW() << "Cannot determine Eltwise post op precision for Convolution node with name '" << getName() << "'";
    }

    return eltwisePrecision;
}

void MKLDNNConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    withBiases = getOriginalInputsNumber() == 3;

    if (!implPriorities.empty()) {
        isPrimitivesPriorityDefined = true;
        // winograd support only constant weights and bias
        isWino = std::find(implPriorities.begin(), implPriorities.end(), impl_desc_type::jit_avx512_winograd) != implPriorities.end() &&
                 mkldnn::impl::cpu::x64::mayiuse(mkldnn::impl::cpu::x64::avx512_common) && !canBeExecutedInInt8() &&
                 getParentEdgeAt(1)->getParent()->isConstant() && getParentEdgeAt(1)->getParent()->getType() == Input &&
                 (withBiases ? (getParentEdgeAt(2)->getParent()->isConstant() && getParentEdgeAt(2)->getParent()->getType() == Input) : true);
    }

    int expectedInputEdgesNum = static_cast<int>(getOriginalInputsNumber());
    for (int i = 0; i < fusedWith.size(); i++) {
        if (fusedWith[i]->getType() == Convolution) {
            expectedInputEdgesNum += static_cast<int>(fusedWith[i]->getOriginalInputsNumber()) - 1;
        }

        if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                expectedInputEdgesNum++;
            }
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!inputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
    eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
        eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
    }

    // We need to make sure that convolution output and second input of fused Eltwise operation
    // have equal precision sizes since they use the same physical memory. In case precisions are different we upscale to FP32.
    if (outputDataType != memory::data_type::f32 && outputDataType != memory::data_type::bf16 && withSum) {
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    if (MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType).size() != eltwisePrecision.size()) {
                        eltwisePrecision = Precision::FP32;
                        outputDataType = memory::data_type::f32;
                    }
                    break;
                }
            }
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
        IE_THROW() << "Incorrect number of input edges for layer " << getName() << ", expected: " << expectedInputEdgesNum
                   << " actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    int ndims = getInputShapeAtPort(0).getRank();

    withDWConv = isFusedWith(Convolution);
    if (withDWConv && isDynamicNode()) {
        IE_THROW() << "DW convolution is fused into convolution node " << getName() << " with dynamic shape.";
    }

    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
        if (convolutionNode) {
            auto& inActivationDims = convolutionNode->inputShapes[0].getStaticDims();
            dw_conv_ih = inActivationDims[convolutionNode->inputShapes[0].getRank() - 2];
            dw_conv_iw = inActivationDims[convolutionNode->inputShapes[0].getRank() - 1];

            auto& outDims = convolutionNode->outputShapes[0].getStaticDims();
            dw_conv_oc = outDims[1];

            const auto &dwWeightsDims = convolutionNode->inputShapes[1].getStaticDims();
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 1]);
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 2]);
            dw_conv_strides = convolutionNode->getStride();

            if (canBeExecutedInInt8()) {
                if (i == 0) {
                    dw_conv_in_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
                } else {
                    dw_conv_in_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[i - 1]->getOriginalOutputPrecisionAtPort(0));
                }
            } else {
                dw_conv_in_dt = memory::data_type::f32;
            }

            for (int j = 0; j < paddingR.size(); j++) {
                int with_group = isGrouped ? 1 : 0;
                int krn = weightDims[with_group + 2 + j];
                int src = getInputShapeAtPort(0).getStaticDims()[2 + j];
                int dst = getOutputShapeAtPort(0).getStaticDims()[2 + j];

                krn = (krn - 1)*(dilation[j] + 1) + 1;
                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
                paddingR[j] = (dst - calc_dst) * stride[j];
            }
        }
    }

    MemoryDescPtr in_candidate, out_candidate;
    if (canBeExecutedInInt8()) {
        //  We have to extend convolution_x8s8s32x from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        if (eltwisePrecision == Precision::BF16)
            eltwisePrecision = Precision::FP32;
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), inputDataType,
            ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), outputDataType,
            ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        inputDataType = (getOriginalInputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
        outputDataType = (getOriginalOutputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
        eltwisePrecision = Precision::FP32;
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    // TODO(amalyshe): there might be situation when convolution can be executed in BF16,
                    // output is required in FP32 but eltwise inplace tensor would be in BF16
                    // currently we forcedly change output to the BF16 that will add reoreder after the node
                    // Another situation can be when we mark output as FP32 and Eltwise asPrecison (which stand
                    // for input of inplace tensor precision) to FP32. This will add reorder for that in-place tensor
                    // bofore the fused convolution. This behaviour might be more correct regarding expected markup
                    // of the graph but performance of first and second approaches might be different. Need to verify
                    outputDataType = eltwisePrecision == Precision::BF16 ? memory::data_type::bf16 : memory::data_type::f32;
                    eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
                }
            }
        }
        // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
        if (inputDataType == memory::data_type::f32
            && (outputDataType == memory::data_type::bf16 || eltwisePrecision == Precision::BF16)) {
            outputDataType = memory::data_type::f32;
            eltwisePrecision = Precision::FP32;
        }

        if (one_of(ndims, 3, 4, 5)) {
            memory::format_tag nspc = ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc);
            memory::format_tag ncsp = ndims == 3 ? memory::format_tag::ncw : (ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw);
            memory::format_tag nCsp8c = ndims == 3 ? memory::format_tag::nCw8c : (ndims == 4 ? memory::format_tag::nChw8c : memory::format_tag::nCdhw8c);
            memory::format_tag nCsp16c = ndims == 3 ? memory::format_tag::nCw16c : (ndims == 4 ? memory::format_tag::nChw16c : memory::format_tag::nCdhw16c);

            auto inputShape = getInputShapeAtPort(0);
            auto outputShape = getOutputShapeAtPort(0);

            if (one_of(inputDataType, memory::data_type::f32, memory::data_type::bf16) &&
                    impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
                createDescriptor({ in_candidate }, { out_candidate });
            }

            if (IC == 1 && groupOC == 1) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
                createDescriptor({ in_candidate }, { out_candidate });
            } else if (IC < 4) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
                createDescriptor({ in_candidate }, { out_candidate });
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
                createDescriptor({ in_candidate }, { out_candidate });
            } else {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp16c);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
                createDescriptor({ in_candidate }, { out_candidate });
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp8c);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
                createDescriptor({ in_candidate }, { out_candidate });
            }

            in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
            out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
            createDescriptor({ in_candidate }, { out_candidate });

            if ((inputDataType != memory::data_type::bf16 && isNspcAvailable()) ||
                    (one_of(inputDataType, memory::data_type::f32, memory::data_type::bf16) &&
                    impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core))) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
                createDescriptor({ in_candidate }, { out_candidate });
            }
        }
    }
}

void MKLDNNConvolutionNode::setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims, bool initWeights = false) {
    mkldnn::post_ops ops;
    const bool useLegacyPostOps = true; // @todo remove after issue with performance of binary post ops fixed

    auto getBinPostOpShape = [&](){
        const auto outShape = getOutputShapeAtPort(0).getStaticDims();
        const auto outShapeRank = getOutputShapeAtPort(0).getRank();
        const auto chIdx = getFusingAxis();
        std::vector<size_t> binaryShape(outShapeRank, 1);
        binaryShape[chIdx] = outShape[chIdx];
        return binaryShape;
    };

    for (auto &node : fusedWith) {
        if (node->getType() == Split || node->getType() == Concatenation)
            continue;

        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            if (eltwiseNode->isSpecialConvolutionAddFusing()) {
                if (withSumBroadcast) {
                    break;
                }
                ops.append_sum(1.0, MKLDNNExtensionUtils::IEPrecisionToDataType(eltwisePrecision));
            } else {
                if (useLegacyPostOps || eltwiseNode->getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
                    eltwiseNode->appendPostOps(ops, dims, postOpsArgs);
                } else {
                    eltwiseNode->appendBinPostOps(ops, getBinPostOpShape(), postOpsArgs);
                }
            }
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            if (useLegacyPostOps) {
                fakeQuantizeNode->appendPostOps(ops, dims, postOpsArgs);
            } else {
                fakeQuantizeNode->appendBinPostOps(ops, getBinPostOpShape(), postOpsArgs);
            }
            continue;
        }

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            if (initWeights) {
                postOpsArgs.push_back(getParentEdgeAt(getOriginalInputsNumber() + 0)->getMemoryPtr());
                postOpsArgs.push_back(getParentEdgeAt(getOriginalInputsNumber() + 1)->getMemoryPtr());

                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                   mkldnn::memory::convert_to_c(dw_conv_in_dt));
            } else {
                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                   mkldnn::memory::convert_to_c(dw_conv_in_dt));
            }
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNConvolutionNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}

void MKLDNNConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // attr[0] - depthwise, quantize
    // attr[1] - binary
    mkldnn::primitive_attr attrs[1];
    setPostOps(attrs[0], MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0)).getStaticDims());

    bool containJitImpl = false;

    for (auto& desc : descs) {
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        for (auto &attr : attrs) {
            addZeroPoints(attr);
            auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            while (static_cast<bool>(itpd)) {
                NodeConfig config;
                config.dynBatchSupport = true;
                for (size_t i = 0; i < descInputNumbers(desc); i++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    auto desc = getSrcMemDesc(itpd, i);
                    if (desc->getType() & MemoryDescType::Blocked && !isGrouped) {
                        dataConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        dataConfig.setMemDesc(std::move(desc));
                    }

                    config.inConfs.push_back(dataConfig);
                }

                if (withDWConv) {
                    auto weightsPrc = MKLDNNExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                    auto biasPrc = memory::data_type::f32;

                    std::vector<size_t> dwWeightsDims({dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    std::vector<size_t> dwBiasesDims({dw_conv_oc});

                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims), weightsPrc, memory::format_tag::Goihw8g));
                    config.inConfs.push_back(dataConfig);

                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), biasPrc, memory::format_tag::x));
                    config.inConfs.push_back(dataConfig);
                 }

                for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                    PortConfig dataConfig;
                    if (withSum) {
                        dataConfig.inPlace(getParentEdges().size() - 1);
                    }

                    dataConfig.constant(false);
                    auto desc = getDstMemDesc(itpd, i);
                    if (desc->getType() & MemoryDescType::Blocked && !isGrouped) {
                        dataConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        dataConfig.setMemDesc(std::move(desc));
                    }

                    config.outConfs.push_back(dataConfig);

                    if (withSum) {
                        dataConfig.inPlace(-1);
                        dataConfig.setMemDesc(getSumMemDesc(itpd)->cloneWithNewPrecision(dataConfig.getMemDesc()->getPrecision()));
                        config.inConfs.push_back(dataConfig);
                    }
                }
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
                if (impl_type & jit)
                    containJitImpl = true;

                supportedPrimitiveDescriptors.emplace_back(config, impl_type);
                if (!itpd.next_impl())
                    break;
            }
        }
    }
}

bool MKLDNNConvolutionNode::created() const {
    return getType() == Convolution;
}

namespace {
std::shared_ptr<mkldnn::convolution_forward::desc>
createDescriptorInternal(const mkldnn::memory::desc& inputDesc,
                         const mkldnn::memory::desc& weightDesc,
                         const mkldnn::memory::desc& biasDesc,
                         const mkldnn::memory::desc& outputDesc,
                         bool withBiases,
                         const std::vector<size_t>& stride,
                         const std::vector<ptrdiff_t>& dilation,
                         const std::vector<ptrdiff_t>& paddingL,
                         const std::vector<ptrdiff_t>& paddingR,
                         mkldnn::algorithm alg) {
    std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
    try {
        if (withBiases) {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          inputDesc, weightDesc, biasDesc, outputDesc,
                                                          mkldnn::memory::dims(stride.begin(), stride.end()),
                                                          mkldnn::memory::dims(dilation.begin(), dilation.end()),
                                                          mkldnn::memory::dims(paddingL.begin(), paddingL.end()),
                                                          mkldnn::memory::dims(paddingR.begin(), paddingR.end())));
        } else {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          inputDesc, weightDesc, outputDesc,
                                                          mkldnn::memory::dims(stride.begin(), stride.end()),
                                                          mkldnn::memory::dims(dilation.begin(), dilation.end()),
                                                          mkldnn::memory::dims(paddingL.begin(), paddingL.end()),
                                                          mkldnn::memory::dims(paddingR.begin(), paddingR.end())));
        }
    } catch (...) {
        IE_THROW() << "Cannot create convolution forward descriptor";
    }
    return conv_desc;
}
} // namespace

void MKLDNNConvolutionNode::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                             const std::vector<MemoryDescPtr>& outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        auto dummyInDims = MemoryDescUtils::makeDummyShape(inputDesc[0]->getShape()).getStaticDims();
        dummyInDims[1] = IC;
        inpDesc = inputDesc[0]->cloneWithNewDims(dummyInDims);
    }
    DnnlMemoryDescPtr definedInpMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc);
    DnnlMemoryDescPtr definedOutMemDesc;

    if (outputDesc[0]->isDefined()) {
        definedOutMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0]);
    } else {
        std::vector<Shape> shapes = { definedInpMemDesc->getShape(), Shape(weightDims) };
        auto outDims = shapeInferGeneric(shapes);
        definedOutMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0]->cloneWithNewDims(outDims.front()));
    }

    const auto& inDnnlDesc = definedInpMemDesc->getDnnlDesc();
    const auto& outDnnlDesc = definedOutMemDesc->getDnnlDesc();

    memory::data_type wdt = static_cast<memory::data_type>(inDnnlDesc.data.data_type);

    if (inDnnlDesc.data.data_type == mkldnn_s8 || inDnnlDesc.data.data_type == mkldnn_u8) {
        wdt = memory::data_type::s8;
    }

    mkldnn::memory::desc weightDnnlDesc(MKLDNNExtensionUtils::convertToDnnlDims(weightDims), wdt, memory::format_tag::any);
    mkldnn::memory::desc biasDnnlDesc;

    if (withBiases) {
        memory::data_type bdt = memory::data_type::f32;
        biasDnnlDesc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(biasesDims), bdt, memory::format_tag::any);
    }

    std::vector<mkldnn::algorithm> algorithms;

    if (isWinograd())
        algorithms.push_back(mkldnn::algorithm::convolution_winograd);
    algorithms.push_back(mkldnn::algorithm::convolution_direct);

    updatePadding();
    for (auto alg : algorithms) {
        descs.emplace_back(createDescriptorInternal(inDnnlDesc, weightDnnlDesc, biasDnnlDesc, outDnnlDesc, withBiases,
                                                    stride, dilation, paddingL, paddingR, alg));
    }
}

void MKLDNNConvolutionNode::addZeroPoints(mkldnn::primitive_attr& attr) {
    if (!inputZeroPoints.empty()) {
        attr.set_input_zero_points(inputZeroPoints.size(), 1 << 1 /*through C dim*/);

        if (!inputZeroPointsMemPtr) {
            inputZeroPointsMemPtr.reset(new MKLDNNMemory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::U8, {inputZeroPoints.size()});
            inputZeroPointsMemPtr->Create(memoryDesc, inputZeroPoints.data());
        }
    }

    if (!weightsZeroPoints.empty()) {
        attr.set_weights_zero_points(weightsZeroPoints.size(), 1 << 1 /*through C dim*/);

        if (!weightsZeroPointsMemPtr) {
            weightsZeroPointsMemPtr.reset(new MKLDNNMemory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::FP32, {weightsZeroPoints.size()});
            weightsZeroPointsMemPtr->Create(memoryDesc, weightsZeroPoints.data());
        }
    }

    if (!outputCompensation.empty()) {
        attr.set_output_compensations(outputCompensation.size(), 1 << 1 /*through C dim*/);

        if (!outputCompensationMemPtr) {
            outputCompensationMemPtr.reset(new MKLDNNMemory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::I32, {outputCompensation.size()});
            outputCompensationMemPtr->Create(memoryDesc, outputCompensation.data());
        }
    }
}

void MKLDNNConvolutionNode::initDescriptor(const NodeConfig& config) {
    auto *selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    // Strided blobs feature support.
    // Works only for FP32 convolutions for now.
    bool isStridedBlobsSupported = true;

    // TODO [NM]: refactor via using global executionPrecision.
    if (canBeExecutedInInt8()) {
        isStridedBlobsSupported = false;
    }

    if (isStridedBlobsSupported) {
        createDescriptor({config.inConfs[0].getMemDesc()}, {config.outConfs[0].getMemDesc()});
    }
    // attr[0] - depthwise, quantize
    // attr[1] - binary
    mkldnn::primitive_attr attrs[1];
    setPostOps(attrs[0], MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0)).getStaticDims());

    auto rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;

    bool containJitImpl = false;

    for (size_t i = 0; i < descs.size(); i++) {
        auto& desc = descs[i];
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        for (auto &attr : attrs) {
            addZeroPoints(attr);
            auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            while (static_cast<bool>(itpd)) {
                NodeConfig cfg;
                cfg.dynBatchSupport = true;
                for (size_t j = 0; j < descInputNumbers(desc); j++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(getSrcMemDesc(itpd, j));
                    cfg.inConfs.push_back(dataConfig);
                }

                if (withDWConv) {
                    auto weightsPrc = MKLDNNExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                    auto biasPrc = memory::data_type::f32;

                    std::vector <size_t> dwWeightsDims({dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    std::vector <size_t> dwBiasesDims({dw_conv_oc});

                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims), weightsPrc, memory::format_tag::Goihw8g));
                    cfg.inConfs.push_back(dataConfig);

                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), biasPrc, memory::format_tag::x));
                    cfg.inConfs.push_back(dataConfig);
                }

                for (size_t j = 0; j < descOutputNumbers(desc); j++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(getDstMemDesc(itpd, j));
                    if (withSum) {
                        auto eltwiseConfig = dataConfig;
                        eltwiseConfig.setMemDesc(eltwiseConfig.getMemDesc()->cloneWithNewPrecision(eltwisePrecision));
                        cfg.inConfs.push_back(eltwiseConfig);
                        dataConfig.inPlace(getParentEdges().size() - 1);
                    }

                    cfg.outConfs.push_back(dataConfig);
                }
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
                if (impl_type & jit)
                    containJitImpl = true;

                if (selected_count == selectedPrimitiveDescriptorIndex) {
                    if (impl_type != selectedPD->getImplementationType()) {
                        IE_THROW() << "Cannot get the original layer configuration!";
                    }
                    rightConfig = cfg;
                }
                if (i == descs.size() - 1 && isStridedBlobsSupported) {
                    if (impl_type == selectedPD->getImplementationType()) {
                        rightConfig = config;
                    }
                }
                selected_count++;
                if (!itpd.next_impl())
                    break;
            }
        }
    }
    selectedPD->setConfig(rightConfig);
}

void MKLDNNConvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    // We also need to filter descs in Convolution node
    filterSupportedDescriptors();
}

void MKLDNNConvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Convolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                auto src_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.src_desc);
                isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
            }
            if (!outputMemoryFormatsFilter.empty()) {
                auto dst_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.dst_desc);
                isSuitableDesc &= dst_tdesc->isSame(outputMemoryFormatsFilter[0]);
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNConvolutionNode::isPossibleToSkipInitConfig(MKLDNNDescriptor &desc) const {
    //  WA: In some cases, we can predict in advance the type of primitive that will be called in the future.
    //  In particular, isPossibleToSkipInitConfig() checks whether we can skip the creation of primitives with
    //  gemm implementation, which significantly increase the network load time.
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty())
        return false;

    if (isPrimitivesPriorityDefined)
        return false;

    //  Here we check that we will not delete jit_planar_conv primitive by mistake.
    //  It requires:
    //      1) strides equal 1;
    //      2) not grouped;
    //      3) first dim of weights is not 1.
    bool isPossibleJitPlanar = true;
    if (isGrouped || weightDims[0] != 1)
        isPossibleJitPlanar = false;
    for (int i = 0; i < stride.size(); i++)
        if (stride[i] != 1)
            isPossibleJitPlanar = false;

    std::shared_ptr<mkldnn::convolution_forward::desc> convDesc(desc);
    auto srcMemDesc = MKLDNNExtensionUtils::makeDescriptor(convDesc->data.src_desc);
    auto dstMemDesc = MKLDNNExtensionUtils::makeDescriptor(convDesc->data.dst_desc);
    auto srcDataType = convDesc->data.src_desc.data_type;
    auto dstDataType = convDesc->data.dst_desc.data_type;
    bool isPlanarFloatConv = srcMemDesc->hasLayoutType(LayoutType::ncsp)
                             && dstMemDesc->hasLayoutType(LayoutType::ncsp)
                             && srcDataType == memory::data_type::f32
                             && dstDataType == memory::data_type::f32;

    return !isPossibleJitPlanar && isPlanarFloatConv;
}

std::shared_ptr<MemoryDesc> MKLDNNConvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

bool MKLDNNConvolutionNode::canFuse(const MKLDNNNodePtr& node) const {
    return canFuseSimpleOperation(node);
}

mkldnn::memory MKLDNNConvolutionNode::getWeights() const {
    return getParentEdgeAt(1)->getMemory().GetPrimitive();
}

void MKLDNNConvolutionNode::setDynamicBatchLim(int lim) {
    if (!execPtr) {
        IE_THROW() << "Can't set dynamic batch for Convolution node with name: " << getName() << ", because executor is not compiled";
    }
    if (execPtr->needReordering()) {
        IE_THROW() << "Can't execute Convolution node with dynamic batch via executor with reorders";
    }
    MKLDNNNode::setDynamicBatchLim(lim);
}

mkldnn::memory MKLDNNConvolutionNode::getBias() const {
    return getParentEdgeAt(2)->getMemory().GetPrimitive();
}

InferenceEngine::Precision MKLDNNConvolutionNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

bool MKLDNNConvolutionNode::isNspcAvailable() const {
    using impl::cpu::x64::mayiuse;

    // do not use in non-quantized networks until it is enforced externally
    if (!isInQuantizedGraph) {
        auto predicate = [](memory::format_tag tag) {
            return one_of(tag, memory::format_tag::nwc, memory::format_tag::nhwc, memory::format_tag::ndhwc);
        };
        if (std::none_of(inputMemoryFormatsFilter.begin(), inputMemoryFormatsFilter.end(), predicate)) {
            return false;
        }
    }

    // A bunch of heuristics are designed to cut off not optimal nspc convolution applications
    auto inpDims = getInputShapeAtPort(0).getDims();
    auto outDims = getOutputShapeAtPort(0).getDims();
    auto ndims = inpDims.size();

    if (isDepthWise()) {
        // 1d equivalent cases are painfully slow
        if (inpDims.size() == 3 || 1 == inpDims[inpDims.size() - 2]) {
            return false;
        }
    } else {
        // it was empirically observed that the nspc convolutions perform much slower than the blocked ones if the channels number more than the specific value
        size_t spatialRank = ndims - 2; //two means batch dim plus channels dim

        bool is1x1 = false;

        if (!isGrouped) {
            auto weightDimsReversItr = weightDims.crbegin();
            auto strideReversItr = stride.crbegin();
            auto paddingLreversItr = paddingL.crbegin();
            auto paddingRreversItr = paddingR.crbegin();

            for (size_t i = 0; i < spatialRank; ++i) {
                is1x1 = true
                        && *(weightDimsReversItr++) == 1
                        && *(strideReversItr++) == 1
                        && *(paddingLreversItr++) == 0
                        && *(paddingRreversItr++) == 0;
            }
        }

        // if the activation field size is 1x1 the avx512 1x1 nspc convolution pollutes caches so that the layer after the convolution performs slow
        if (mayiuse(impl::cpu::x64::avx512_common) && is1x1) {
            auto end = inpDims.rbegin();
            std::advance(end, spatialRank);
            if (std::all_of(inpDims.rbegin(), end, [](size_t x) { return dimsEqualStrong(1, x); })) {
                return false;
            }
        }

        unsigned thresholdNumChannels = 128u; // for avx and below
        if (is1x1) {
            thresholdNumChannels = 2048u;
        } else if (mayiuse(impl::cpu::x64::avx512_common)) {
            thresholdNumChannels = 512u;
        }

        size_t OC = outDims[1];
        if (std::max(IC, OC) >= thresholdNumChannels) {
            return false;
        }
        if (!mayiuse(impl::cpu::x64::avx)) {
            // SSE41 nspc convolutions do not support ic and oc tails yet and the blocked implementation will be much better than gemm
            if ((IC % 8) || (OC % 8)) {
                return false;
            }
        }
    }

    return true;
}

InferenceEngine::Blob::Ptr MKLDNNConvolutionNode::createInternalBlob(InferenceEngine::SizeVector dims, size_t edgeNum, bool isGrouped) {
    const auto constNode = std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgeAt(edgeNum)->getParent());
    if (!constNode) {
        IE_THROW() << "Cannot cast " << edgeNum << " input to Input node for " << getName() << ".";
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto const elementsCount = blb->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, dims, getWeightsLayoutByDims(dims, isGrouped));

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    internalBlob->allocate();

    if (internalBlob->size() != elementsCount) {
        IE_THROW() << "Created internal blob and const blob has different size for node: " << getName() << ".";
    }

    cpu_convert(blb->GetPtr(),
                internalBlob->buffer(),
                MKLDNNExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()),
                internalBlob->getTensorDesc().getPrecision(),
                elementsCount);

    return internalBlob;
}

void MKLDNNConvolutionNode::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto wghMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
    auto dstMemPtr = getOutputMemory();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory was not allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory was not allocated.";
    if (!wghMemPtr || !wghMemPtr->isAllocated())
        IE_THROW() << "Weight memory was not allocated.";
    MKLDNNMemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory didn't allocate.";
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    DnnlMemoryDescCPtr inMemoryDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr weightMemoryDesc = wghMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outMemoryDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr biasDesc;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    auto initPrimitiveAttr = [&]() {
        mkldnn::primitive_attr attr;
        addZeroPoints(attr);
        setPostOps(attr, outMemoryDesc->getShape().getStaticDims(), true);

        return std::make_shared<mkldnn::primitive_attr>(std::move(attr));
    };

    AttrPtr pAttrLocal;

    if (isDynamicNode()) {
        if (!pAttr || withSum) {
            pAttr = initPrimitiveAttr();
        }
        pAttrLocal = pAttr;
    } else {
        pAttrLocal = initPrimitiveAttr();
    }

    updatePadding();
    ConvKey key = {inMemoryDesc,
                   weightMemoryDesc,
                   biasDesc,
                   outMemoryDesc,
                   stride,
                   dilation,
                   paddingL,
                   paddingR,
                   *pAttrLocal,
                   selected_pd->getImplementationType()};

    auto engine = getEngine();
    auto builder = [&engine](const ConvKey& key) -> executorPtr {
        auto createMkldnnConvDesc = [](const mkldnn::memory::desc& srcDesc,
                                       const mkldnn::memory::desc& wghDesc,
                                       const mkldnn::memory::desc& dstDesc,
                                       DnnlMemoryDescCPtr biasDescPtr,
                                       const std::vector<size_t>& stride,
                                       const std::vector<ptrdiff_t>& dilation,
                                       const std::vector<ptrdiff_t>& paddingL,
                                       const std::vector<ptrdiff_t>& paddingR,
                                       mkldnn::algorithm alg) -> std::shared_ptr<MKLDNNDescriptor> {
            mkldnn::memory::desc dnnlBiasDesc;
            if (biasDescPtr) {
                // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
                dnnlBiasDesc = biasDescPtr->getDnnlDesc().reshape({dstDesc.dims()[1]});
            }

            return std::make_shared<MKLDNNDescriptor>(createDescriptorInternal(srcDesc,
                                                                               wghDesc,
                                                                               dnnlBiasDesc,
                                                                               dstDesc,
                                                                               (biasDescPtr != nullptr),
                                                                               stride,
                                                                               dilation,
                                                                               paddingL,
                                                                               paddingR,
                                                                               alg));
        };

        const auto alg = (key.implType & impl_desc_type::winograd) ? mkldnn::algorithm::convolution_winograd : mkldnn::algorithm::convolution_direct;
        std::shared_ptr<MKLDNNDescriptor> desc = createMkldnnConvDesc(key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      key.bias,
                                                                      key.stride,
                                                                      key.dilation,
                                                                      key.paddingL,
                                                                      key.paddingR,
                                                                      alg);

        auto itpd = desc->createPrimitiveDescriptorIterator(engine, key.attr);

        executorPtr execPtr = nullptr;
        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                auto prim_desc = convolution_forward::primitive_desc(itpd.get());
                execPtr = std::make_shared<ConvolutionExecutor>(prim_desc,
                                                                key.inp0->getDnnlDesc(),
                                                                key.inp1->getDnnlDesc(),
                                                                key.out->getDnnlDesc(),
                                                                engine);
                break;
            }

            if (!itpd.next_impl()) {
                break;
            }
        }

        if (!execPtr) {
            auto inDesc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(key.inp0->getShape().getStaticDims()),
                                                                                           key.inp0->getDataType(),
                                                                                           memory::format_tag::any);
            auto wghDesc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
                                                                                        key.inp1->getDataType(),
                                                                                        memory::format_tag::any);
            auto outDesc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
                                                                                        key.out->getDataType(),
                                                                                        memory::format_tag::any);

            std::shared_ptr<MKLDNNDescriptor> reorderConvDesc = createMkldnnConvDesc(inDesc,
                                                                                     wghDesc,
                                                                                     outDesc,
                                                                                     key.bias,
                                                                                     key.stride,
                                                                                     key.dilation,
                                                                                     key.paddingL,
                                                                                     key.paddingR,
                                                                                     mkldnn::algorithm::convolution_direct);

            auto reordItpd = reorderConvDesc->createPrimitiveDescriptorIterator(engine, key.attr);
            if (static_cast<bool>(reordItpd)) {
                auto prim_desc = convolution_forward::primitive_desc(reordItpd.get());
                execPtr = std::make_shared<ConvolutionExecutor>(prim_desc,
                                                                key.inp0->getDnnlDesc(),
                                                                key.inp1->getDnnlDesc(),
                                                                key.out->getDnnlDesc(),
                                                                engine);
            }
        }

        return execPtr;
    };

    execPtr = nullptr;
    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;

    if (execPtr) {
        primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
        primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
        primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        }

        appendZeroPointsArgs();
        MKLDNNNode::appendPostOpArgs(*pAttrLocal, primArgs, postOpsArgs);
    } else {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }
}

MKLDNNConvolutionNode::ConvolutionExecutor::ConvolutionExecutor(const mkldnn::convolution_forward::primitive_desc& pd,
                                                                const mkldnn::memory::desc& inMemDesc,
                                                                const mkldnn::memory::desc& weightMemDesc,
                                                                const mkldnn::memory::desc& outMemDesc,
                                                                const mkldnn::engine& engine) {
    execPrim.reset(new mkldnn::convolution_forward(pd));

    if (inMemDesc != pd.src_desc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, pd.src_desc(), engine)});
    }

    if (weightMemDesc != pd.weights_desc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, pd.weights_desc(), engine)});
    }

    if (outMemDesc != pd.dst_desc()) {
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(pd.dst_desc(), outMemDesc, engine)});
    }
}

void MKLDNNConvolutionNode::execute(mkldnn::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Convolution node with name: " << getName() << ", because executor is not compiled";
    }
    execPtr->exec(primArgs, strm);
}

void MKLDNNConvolutionNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
    if (withSumBroadcast) {
        if (!subgraph) {
            IE_THROW(Unexpected) << "Fused ops subgraph has not been created in " << getTypeStr() << " with name " << getName();
        }
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgesAtPort(sumPortNum).front()->getMemory();
        auto inp1 = subgraph->getInput(1);
        inp1->getChildEdgesAtPort(0).front()->getMemoryPtr()->setDataHandle(sumInpMem.GetData());

        subgraph->infer();

        auto out = subgraph->getOutput(0);
        const auto& outMem = out->getParentEdgesAtPort(0).front()->getMemory();
        auto convOutMem = getChildEdgesAtPort(0).front()->getMemoryPtr();
        convOutMem->redefineDesc(getBaseMemDescAtOutputPort(0)->cloneWithNewDims(outMem.getStaticDims()));
        convOutMem->SetData(outMem);
    }
}

void MKLDNNConvolutionNode::updatePadding() {
    //update padding.
    if (isDynamicNode() && autoPadding) {
        paddingL = shapeInference->get_pads_begin();
        paddingR = shapeInference->get_pads_end();
    }
}

void MKLDNNConvolutionNode::redefineOutputMemory(const std::vector<VectorDims> &newOutputShapes) {
    if (withSum) {
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgesAtPort(sumPortNum).front()->getMemory();
        if (newOutputShapes.front() != sumInpMem.getStaticDims()) {
            withSumBroadcast = true;
            if (!subgraph) {
                subgraph = std::make_shared<FusedSubgraph>(fusedWith, *this, weightCache);
            }
            auto inp0 = subgraph->getInput(0);
            inp0->redefineOutputMemory(newOutputShapes);

            auto inp1 = subgraph->getInput(1);
            inp1->redefineOutputMemory({sumInpMem.getStaticDims()});
            // here we postpone output memory reallocation due to the fact that it is the same memory with the sum second input
            return;
        } else {
            withSumBroadcast = false;
        }
    }
    MKLDNNNode::redefineOutputMemory(newOutputShapes);
}

MemoryDescPtr MKLDNNConvolutionNode::getSumMemDesc(primitive_desc_iterator &primitive_desc_it) {
    if (getOutputShapeAtPort(0).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(primitive_desc_it.dst_desc(0), getInputShapeAtPort(getParentEdges().size() - 1));
    }
    return MKLDNNExtensionUtils::makeDescriptor(primitive_desc_it.dst_desc(0));
}

MKLDNNMemoryPtr MKLDNNConvolutionNode::getOutputMemory() const {
    if (withSumBroadcast) {
        if (!subgraph) {
            IE_THROW(Unexpected) << "Fused ops subgraph has not been created in " << getTypeStr() << " with name " << getName();
        }
        auto inp0 = subgraph->getInput(0);
        return inp0->getChildEdgesAtPort(0).front()->getMemoryPtr();
    } else {
        return getChildEdgesAtPort(0).front()->getMemoryPtr();
    }
}

void MKLDNNConvolutionNode::addFusedNode(const MKLDNNNodePtr &fusingNode) {
    if (Eltwise == fusingNode->getType()) {
        if (fusingNode->getAlgorithm() == EltwiseAdd) {
            auto eltwiseNode = std::dynamic_pointer_cast<MKLDNNEltwiseNode>(fusingNode);
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                withSum = true;
            }
        }
        if (withSum && isDynamicNode()) {
            for (size_t i = 0; i < fusingNode->getParentEdges().size(); ++i) {
                auto edge = fusingNode->getParentEdgesAtPort(i).front();
                auto parent = edge->getParent();
                if ("Constant" == parent->getTypeStr()) {
                    fusedConstNodes[fusingNode].push_back(parent);
                }
            }
        }
    }
    MKLDNNNode::addFusedNode(fusingNode);
}

void MKLDNNConvolutionNode::appendZeroPointsArgs() {
    if (inputZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = inputZeroPointsMemPtr->GetPrimitive();
    }
    if (weightsZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = weightsZeroPointsMemPtr->GetPrimitive();
    }
    if (outputCompensationMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST] = outputCompensationMemPtr->GetPrimitive();
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNConvolutionNode, Convolution);
