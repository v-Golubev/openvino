// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "edge.h"
#include "node.h"
#include "dnnl_extension_utils.h"
#include <blob_factory.hpp>
#include "nodes/input.h"

using namespace dnnl;
namespace ov {
namespace intel_cpu {

Edge::Edge(const NodePtr &parent, const NodePtr &child, int pr_port, int ch_port) :
        parent(parent), child(child), parent_port(pr_port), child_port(ch_port) {}

const NodePtr Edge::getParent() const {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        IE_THROW() << "Edge contains empty parent node";
    return parentPtr;
}

const NodePtr Edge::getChild() const {
    auto childPtr = child.lock();
    if (!childPtr)
        IE_THROW() << "Edge contains empty child node";
    return childPtr;
}

bool Edge::isUseExternalMemory() const {
    return useExternalMemory;
}

bool Edge::isDropped() const {
    bool not_in_parent = true;
    bool not_in_child = true;

    auto parent_ptr = parent.lock();
    if (parent_ptr) {
        for (auto &edge : parent_ptr->childEdges)
            if (edge.lock().get() == this)
                not_in_parent = false;
    }

    auto child_ptr = child.lock();
    if (child_ptr) {
        for (auto &edge : child_ptr->parentEdges)
            if (edge.lock().get() == this)
                not_in_child = false;
    }
    return not_in_parent && not_in_child;
}

void Edge::drop() {
    auto _drop_from = [&] (std::vector<EdgeWeakPtr> &list) {
        auto myself = std::find_if(list.begin(), list.end(),
                [&] (EdgeWeakPtr edge) { return edge.lock().get() == this; });

        if (myself != list.end())
            list.erase(myself);
    };

    _drop_from(getParent()->childEdges);
    _drop_from(getChild()->parentEdges);
}

void Edge::collectConsumers(std::vector<NodePtr>& result) const {
    if (this->inPlace(LOOK_DOWN)) {
        if (auto peerChildSPD = this->getChild()->getSelectedPrimitiveDescriptor()) {
            auto peerOutputNum = this->getOutputNum();
            auto peerInPlacePort = peerChildSPD->getConfig().inConfs[peerOutputNum].inPlace();
            auto& vecChildEdges = this->getChild()->getChildEdgesAtPort(peerInPlacePort);
            for (auto childEdge : vecChildEdges) {
                childEdge->collectConsumers(result);
            }
        }
    } else {
        result.push_back(this->getChild());
    }
}

static bool childCanChangeMem(const Edge& edge) {
    int outNumber = edge.getOutputNum();
    auto child = edge.getChild();
    if (auto childSPD = edge.getChild()->getSelectedPrimitiveDescriptor()) {
        if (childSPD->getConfig().outConfs.empty())
            return true;

        for (int port = 0; port < childSPD->getConfig().outConfs.size(); port++) {
            const auto& conf = childSPD->getConfig().outConfs[port];
            if (conf.inPlace() == outNumber && outNumber >= 0) {
                // WA. In general even if some operation is has inplace config it doesn't mean it will change underlaying memory during inference
                // Example: Split operation which in some cases just creates view on the same tensor
                // In which cases we have to recursivly check childs of such layers
                // TODO: how to understand in general way if node can change memory during inference? Extend Node API?
                if (one_of(child->getType(), Type::Split, Type::Concatenation)) {
                    for (const auto& childEdge : child->getChildEdgesAtPort(port)) {
                        if (childCanChangeMem(*childEdge)) {
                            return true;
                        }
                    }
                } else {
                    return true;
                }
            }
        }
    }
    return false;
}

bool Edge::enforceReorder() {
    bool canBeInPlaceConflicts = false;
    auto parentNode = getParent();
    auto parentSPD = parentNode->getSelectedPrimitiveDescriptor();
    auto childNode = getChild();
    auto childSPD = childNode->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        IE_THROW() << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";

    const auto& detectInPlaceChildrenNum = [](const std::vector<EdgePtr>& edges) -> size_t {
        size_t count = 0;
        for (const auto& edge : edges) {
            if (childCanChangeMem(*edge)) {
                count++;
            }
        }
        return count;
    };

    bool in_place = inPlace();
    int inNumber = getInputNum();

    const auto portChildEdges = parentNode->getChildEdgesAtPort(inNumber);
    if (childCanChangeMem(*this) && portChildEdges.size() > 1) {
        if (childNode->getType() == Type::Convolution) {
            auto execIndex = childNode->getExecIndex();
            for (auto pEdgePeer : portChildEdges) {
                if (pEdgePeer.get() == this)
                    continue;
                std::vector<NodePtr> vecConsumers;
                pEdgePeer->collectConsumers(vecConsumers);

                for (auto node : vecConsumers) {
                    if (node->getExecIndex() >= execIndex) {
                        canBeInPlaceConflicts = true;
                        break;
                    }
                }
                if (canBeInPlaceConflicts) break;
            }
        // TODO: detectInPlaceChildrenNum(portChildEdges) > 1 - is it always safe condition? Looks like depends on exec order
        } else if (in_place && detectInPlaceChildrenNum(portChildEdges) > 1) {
            canBeInPlaceConflicts = true;
        }
    }

    // TODO: looks like can be unified with condition above
    if (!canBeInPlaceConflicts && childCanChangeMem(*this) && !parentNode->getChildEdges().empty()) {
        auto execIndex = childNode->getExecIndex();
        for (auto pEdgePeer : portChildEdges) {
            if (pEdgePeer.get() == this)
                continue;
            std::vector<NodePtr> vecConsumers;
            pEdgePeer->collectConsumers(vecConsumers);

            for (auto node : vecConsumers) {
                if (node->getExecIndex() >= execIndex) {
                    canBeInPlaceConflicts = true;
                    break;
                }
            }
            if (canBeInPlaceConflicts) break;
        }
    }

    if (in_place) {
        int outNumber = getOutputNum();
        if (inNumber >= 0 && inNumber < parentSPD->getConfig().outConfs.size() && parentSPD->getConfig().outConfs[inNumber].inPlace() >= 0 &&
            outNumber >= 0 && outNumber < childSPD->getConfig().inConfs.size() && childSPD->getConfig().inConfs[outNumber].inPlace() >= 0) {
            if (childCanChangeMem(*this)) {
                canBeInPlaceConflicts = true;
            }
        }
    }

    if (canBeInPlaceConflicts) {
        return true;
    }

    // In case the parent node is an input constant, the memory is unaligned and the child primitive isa is SSE,
    // we have to insert reorder since the vast majority of arithmetic and data processing instructions in legacy SSE isa requires
    // the memory address in the operands must be aligned on 16-byte boundary.
    if ((childSPD->getImplementationType() & impl_desc_type::sse42) &&
        Type::Input == parentNode->getType() &&
        parentNode->isConstant()) {
        if (auto pInputNode = std::dynamic_pointer_cast<node::Input>(parentNode)) {
            auto rawMemPtr = pInputNode->getMemoryPtr()->GetData();
            bool isAligned = (reinterpret_cast<uintptr_t>(rawMemPtr) & 15) == 0;
            if (!isAligned) {
                return true;
            }
        }
    }

    return false;
}

bool isPhysicalMemCompatible(const MemoryDesc& lhsMemDesc, const MemoryDesc& rhsMemDesc) {
    if (!lhsMemDesc.isDefined() || !rhsMemDesc.isDefined() ||
        !(lhsMemDesc.getType() & MemoryDescType::Blocked) || !(rhsMemDesc.getType() & MemoryDescType::Blocked) ||
        (lhsMemDesc.getType() == DnnlBlocked && !lhsMemDesc.as<const DnnlMemoryDesc>()->hasEmptyExtraData()) ||
        (rhsMemDesc.getType() == DnnlBlocked && !rhsMemDesc.as<const DnnlMemoryDesc>()->hasEmptyExtraData()))
        return false;

    const auto lhsBlockMemDesc = lhsMemDesc.as<BlockedMemoryDesc>();
    const auto rhsBlockMemDesc = rhsMemDesc.as<BlockedMemoryDesc>();

    if (lhsBlockMemDesc->getShape() != rhsBlockMemDesc->getShape() || lhsBlockMemDesc->getPrecision() != rhsBlockMemDesc->getPrecision())
        return false;

    // dims padding check
    bool isZeroDimsPaddings =
        std::all_of(lhsBlockMemDesc->getOffsetPaddingToData().begin(), lhsBlockMemDesc->getOffsetPaddingToData().end(), [](size_t x){ return x == 0; }) &&
        std::all_of(rhsBlockMemDesc->getOffsetPaddingToData().begin(), rhsBlockMemDesc->getOffsetPaddingToData().end(), [](size_t x){ return x == 0; });
    bool isSameElementsCount = lhsBlockMemDesc->getPaddedElementsCount() == rhsBlockMemDesc->getPaddedElementsCount();
    if (!isZeroDimsPaddings || !isSameElementsCount)
        return false;

    // tensor padding check
    // TODO: why tensors with different offset paddings are not physically compatible. Looks like this condition can be removed
    // if (lhsBlockMemDesc->getOffsetPadding() != rhsBlockMemDesc->getOffsetPadding()) {
    //     return false;
    // }

    // stride check
    const auto lhsBlockDims = lhsBlockMemDesc->getBlockDims();
    std::vector<size_t> lhsStridesDefault(lhsBlockDims.size());
    lhsStridesDefault[lhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= lhsBlockDims.size(); i++) {
        lhsStridesDefault[lhsBlockDims.size() - i] = lhsStridesDefault[lhsBlockDims.size() - (i - 1)] * lhsBlockDims[lhsBlockDims.size() - (i - 1)];
    }

    auto rhsBlockDims = rhsBlockMemDesc->getBlockDims();
    std::vector<size_t> rhsStridesDefault(rhsBlockDims.size());
    rhsStridesDefault[rhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= rhsBlockDims.size(); i++) {
        rhsStridesDefault[rhsBlockDims.size() - i] =
             rhsStridesDefault[rhsBlockDims.size() - (i - 1)] * rhsBlockDims[rhsBlockDims.size() - (i - 1)];
    }

    // this check needed to avoid inserting unnecessary reorders if the memory is used in place and the batch size is equal to 1
    // in nodes like concate and split
    size_t lhsSkipAxis = lhsBlockDims.size() > 0 && lhsBlockDims[0] == 1 ? 0 : Shape::UNDEFINED_DIM;
    size_t rhsSkipAxis = rhsBlockDims.size() > 0 && rhsBlockDims[0] == 1 ? 0 : Shape::UNDEFINED_DIM;

    bool isDenseTensor = dimsEqualStrong(lhsStridesDefault, lhsBlockMemDesc->getStrides(), lhsSkipAxis) &&
                         dimsEqualStrong(rhsStridesDefault, rhsBlockMemDesc->getStrides(), rhsSkipAxis);
    if (!isDenseTensor)
        return false;

    auto getCleanDim = [&](const VectorDims& dims, const VectorDims& flag) {
        if (dims.size() != flag.size())
            return dims;
        std::vector<size_t> ret;
        for (int i = 0; i < dims.size(); i++) {
            if (flag[i] != 1) {
                ret.push_back(dims[i]);
            }
        }
        return ret;
    };

    // block dim check
    auto lhsBlockDimsClean = getCleanDim(lhsBlockDims, lhsBlockDims);
    auto rhsBlockDimsClean = getCleanDim(rhsBlockDims, rhsBlockDims);
    if (!dimsEqualStrong(lhsBlockDimsClean, rhsBlockDimsClean))
        return false;

    // order check
    auto lhsOrderClean = getCleanDim(lhsBlockMemDesc->getOrder(), lhsBlockDims);
    auto rhsOrderClean = getCleanDim(rhsBlockMemDesc->getOrder(), rhsBlockDims);
    if (!dimsEqualStrong(lhsOrderClean, rhsOrderClean))
        return false;

    return true;
}

Edge::ReorderStatus Edge::needReorder() {
    bool optimized = false;
    auto inputPortDesc = getInputPortDesc();
    auto outPortDesc = getOutputPortDesc();
    // Check whether the child node may accept the parent produced tensor
    if (!outPortDesc->isCompatible(*inputPortDesc)) {
        // Performance optimization which exploit the fact that some tensors do not need actual data reordering to be read using different descriptors
        if (isPhysicalMemCompatible(*inputPortDesc->getMemDesc(), *outPortDesc->getMemDesc()) && !getParent()->isConstant()) {
            optimized = true;
        } else {
            return ReorderStatus::Regular;
        }
    }

    // put here as more costly than compatible check
    if (enforceReorder()) {
        return ReorderStatus::Regular;
    }

    if (optimized) {
        return ReorderStatus::Optimized;
    }

    return ReorderStatus::No;
}

void Edge::reuse(MemoryPtr ptr) {
    if (status != Status::NeedAllocation)
        return;
    memoryPtr = ptr;
    status = Status::Allocated;

    DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
}

int Edge::getInputNum() const {
    return parent_port;
}

int Edge::getOutputNum() const {
    return child_port;
}

void Edge::allocateCommon(const std::function<void(const MemoryPtr&, const MemoryDesc&)>& allocate) {
    if (status != Status::NeedAllocation)
        return;

    if (memoryPtr)
        IE_THROW() << "Unexpected behaviour: status == NeedAllocation but memory is already allocated.";

    auto& inputDesc = getInputDesc();
    auto& outputDesc = getOutputDesc();
    if (!inputDesc.isCompatible(outputDesc))
        IE_THROW() << "Cannot allocate memory for incompatible descriptors.";

    auto parentPtr = getParent();
    memoryPtr.reset(new Memory(parentPtr->getEngine()));

    allocate(memoryPtr, inputDesc);
    DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
    status = Status::Allocated;
}

void Edge::allocate(const void* mem_ptr) {
    auto allocateFunc = [=](const MemoryPtr& memoryPtr, const MemoryDesc& inputDesc) {
        memoryPtr->Create(inputDesc, mem_ptr, false);  // no pads zeroing
    };

    allocateCommon(allocateFunc);
}

void Edge::allocate(DnnlMemoryMngrPtr memMngr) {
    if (!memMngr) {
        IE_THROW(Unexpected) << "Memory manager ptr is NULL";
    }

    auto allocateFunc = [=](const MemoryPtr& memoryPtr, const MemoryDesc& inputDesc) {
        memoryPtr->Create(inputDesc, memMngr);
    };

    allocateCommon(allocateFunc);
}

std::string Edge::name() const {
    auto parentPtr = getParent();
    auto childPtr = getChild();

    std::stringstream result;

    result << parentPtr->getName() << " port " << parent_port << " <-> " << childPtr->getName() << " port " << child_port;

    return  result.str();
}

void Edge::externalAllocate(WeightsSharing::Ptr weightsCache) {
    if (status != Status::NeedAllocation)
        return;

    if (weightsCache) {
        auto alloc = [this] () {
            allocate();
            return memoryPtr;
        };

        auto ptr = weightsCache->findOrCreate(name(), alloc, false);
        memoryPtr = *ptr;
        DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
        useExternalMemory = true;
        status = Status::Allocated;
    } else {
        allocate();
    }
}

void Edge::changeStatus(Edge::Status state) {
    if (state == Status::NotAllocated) {
        IE_THROW() << "Incorrect behaviour! Use method sharedMemFrom()";
    }
    if (state == Status::Validated) {
        IE_THROW() << "Incorrect behaviour! Use method validate()";
    }
    if (status != Status::Uninitialized && state == Status::NeedAllocation)
        return;
    if (status == Status::NotAllocated)
        memoryFromEdge.reset();
    status = state;
}

PortDescBaseCPtr Edge::getInputPortDesc() const {
    auto parentPtr = getParent();
    if (parentPtr->getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Primitive descriptor for node " << parentPtr->getName() << " is not selected.";

    int inputIdx = getInputNum();
    if (inputIdx < 0)
        IE_THROW() << "Edge cannot be found for node" << parentPtr->getName() << ".";

    auto& outConfs = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs;
    if (outConfs.empty())
        IE_THROW() << "Node " << parentPtr->getName() << " has empty output config list.";

    if (inputIdx >= outConfs.size())
        inputIdx = 0;

    auto inputPortDesc = outConfs[inputIdx].getPortDesc();
    if (!inputPortDesc) {
        IE_THROW() << "Node" << parentPtr->getName() << " has unitialized input port desc on port " << inputIdx;
    }

    return inputPortDesc;
}

PortDescBaseCPtr Edge::getOutputPortDesc() const {
    auto childPtr = getChild();

    if (childPtr->getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Primitive descriptor for node " << childPtr->getName() << " is not selected.";

    int outputIdx = getOutputNum();
    if (outputIdx < 0) {
        IE_THROW() << "Edge cannot be found for node" << childPtr->getName() << ".";
    }
    auto& inConfs = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs;
    if (inConfs.empty())
        IE_THROW() << "Node " << childPtr->getName() << " has empty input config list.";

    if (outputIdx >= inConfs.size())
        outputIdx = 0;

    auto outPortDesc = inConfs[outputIdx].getPortDesc();
    if (!outPortDesc) {
        IE_THROW() << "Node" << childPtr->getName() << " has unitialized output port desc on port " << outputIdx;
    }

    return outPortDesc;
}

const MemoryDesc& Edge::getInputDesc() const {
    auto memDescPtr = getInputPortDesc()->getMemDesc();
    if (!memDescPtr) {
        IE_THROW() << "Cannot get input memory descriptor for edge: " << getParent()->getName() << "->"
                   << getChild()->getName();
    }
    return *memDescPtr;
}

const MemoryDesc& Edge::getOutputDesc() const {
    auto memDescPtr = getOutputPortDesc()->getMemDesc();
    if (!memDescPtr) {
        IE_THROW() << "Cannot get output memory descriptor for edge: " << getParent()->getName() << "->"
                   << getChild()->getName();
    }
    return *memDescPtr;
}

const MemoryDesc& Edge::getDesc() const {
    if (!getInputDesc().isCompatible(getOutputDesc()))
        IE_THROW() << "Cannot get descriptor for edge: " << getParent()->getName() << "->"
                   << getChild()->getName();

    return getInputDesc();
}

const Memory &Edge::getMemory() {
    return *getMemoryPtr();
}

MemoryPtr &Edge::getMemoryPtr() {
    if (status == Status::NotAllocated) {
        memoryPtr.reset(new Memory(getParent()->getEngine()));
        const auto &desc = getDesc();
        auto sharedEdge = getSharedEdge();
        auto sharedEdgeParent = sharedEdge->getParent();
        if (sharedEdgeParent->isConstant()) {
            memoryPtr->Create(desc, sharedEdge->getMemoryPtr()->GetData());
            DEBUG_LOG(*this, " const sharedEdge with ", *sharedEdge);
        } else {
            memoryPtr->Create(desc, sharedEdge->getMemoryPtr()->getDnnlMemoryMngr());
            DEBUG_LOG(*this, " sharedEdge with ", *sharedEdge);
        }
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return memoryPtr;
}

void Edge::sharedMemFrom(const EdgePtr &edge) {
    memoryFromEdge = edge;
    DEBUG_LOG(*this, " sharedMemFrom ", *edge);
    status = Status::NotAllocated;
}

void Edge::validate() {
    if (status == Status::Validated)
        return;
    getMemory();
    getParent();
    getChild();

    if (status != Status::Allocated) {
        IE_THROW() << "Error memory is not allocated!";
    }
    status = Status::Validated;
}

EdgePtr Edge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    if (!memoryFromEdgePtr) {
        IE_THROW() << "Cannot get memory ptr for edge( " << name() << " ). The pointer on the edge with memory is empty!";
    }
    return memoryFromEdgePtr;
}

EdgePtr Edge::getSharedEdge(std::nothrow_t) const {
    return memoryFromEdge.lock();
}

void Edge::init() {
    if (status != Status::NeedAllocation && status != Status::Uninitialized)
        return;
    DEBUG_LOG(*this);
    EdgePtr edgePtr = getBaseEdge();
    if (edgePtr.get() == this) {
        DEBUG_LOG(*this, " getBaseEdge() return itself");
        changeStatus(Status::NeedAllocation);
    } else {
        if (edgePtr->getParent()->isConstant() && !edgePtr->getChild()->isConstant()) {
            changeStatus(Status::NeedAllocation);
            DEBUG_LOG(*this, " edge inplace from ", *edgePtr, " is broken!");
            return;
        }
        sharedMemFrom(edgePtr);
    }

    auto port = getInputNum();
    if (port < 0)
        return;
    auto edges_at_same_port = getParent()->getChildEdgesAtPort(static_cast<size_t>(port));
    for (auto edge : edges_at_same_port) {
        if (edge->getStatus() != Status::NeedAllocation && edge->getStatus() != Status::Uninitialized) {
            if (edge->getSharedEdge() != edgePtr)
                IE_THROW() << "Unsupported behavior. Cannot mark edge "
                                   << getParent()->getChildEdgeAt(0)->getParent()->getName() << "->"
                                   << getParent()->getChildEdgeAt(0)->getChild()->getName() << " as not allocated!";
        } else {
            if (edge != edgePtr)
                edge->sharedMemFrom(edgePtr);
        }
    }
}

/**
 * Should analyze graph node dependencies, inplace node information and return root memory(edge) it view on
 *
 * @param type some magic enum values... description needed
 * @return root of view-on-memory subgraph
 */
EdgePtr Edge::getBaseEdge(int look) {
    auto parentConfig = getParent()->getSelectedPrimitiveDescriptor()->getConfig();
    auto childConfig = getChild()->getSelectedPrimitiveDescriptor()->getConfig();
    int inputNum = getInputNum();
    int outputNum = getOutputNum();

    if (childConfig.inConfs[outputNum].inPlace() >= 0 && parentConfig.outConfs[inputNum].inPlace() >= 0) {
        // in case of parentConfig requiring upstream-inplace and childConfig supports downstream-inplace
        // must further check whether childConfig also supports upstream inplace,
        // if so, we can safely inplace as upstream
        auto down_stream_inplace = childConfig.inConfs[outputNum].inPlace();
        int up_stream_inplace = -1;
        if (down_stream_inplace >= 0)
            up_stream_inplace = childConfig.outConfs[down_stream_inplace].inPlace();

        if ((up_stream_inplace >= 0) && (look & LOOK_UP)) {
            look = LOOK_UP;
        } else {
            DEBUG_LOG(*this, " Danger: Inplace assumption will be broken!");
            inputNum = getInputNum();
            return getParent()->getChildEdgeAt(inputNum);
        }
    }

    if (childConfig.inConfs[outputNum].inPlace() >= 0 && (look & LOOK_DOWN)) {
        int next_port_idx = childConfig.inConfs[outputNum].inPlace();
        if (childConfig.outConfs[next_port_idx].inPlace() >= 0) {
            childConfig.outConfs[next_port_idx].inPlace(-1);
            getChild()->initDescriptor(childConfig);
        }

        auto ch_edges = getChild()->getChildEdgesAtPort(next_port_idx);
        auto &next_ch_edge = ch_edges[0];

        // Multiple connection to some out port
        // Will try to find inplace consumer
        for (auto &ch_edge : ch_edges) {
            auto &chch_conf = ch_edge->getChild()->getSelectedPrimitiveDescriptor()->getConfig();

            if (chch_conf.inConfs[ch_edge->getOutputNum()].inPlace() >= 0)
                next_ch_edge = ch_edge;
        }
        return next_ch_edge->getBaseEdge(LOOK_DOWN);
    } else if (parentConfig.outConfs[inputNum].inPlace() >= 0 && (look & LOOK_UP)) {
        int next_port_idx = parentConfig.outConfs[inputNum].inPlace();
        if (parentConfig.inConfs[next_port_idx].inPlace() >= 0) {
            parentConfig.inConfs[next_port_idx].inPlace(-1);
            getParent()->initDescriptor(parentConfig);
        }
        return getParent()->getParentEdgesAtPort(next_port_idx)[0]->getBaseEdge(LOOK_UP);
    }

    auto edges_for_same_port = getParent()->getChildEdgesAtPort(inputNum);
    if (!(look & LOOK_NO_RECURRENT)) {
        for (auto edge : edges_for_same_port) {
            if (edge.get() != this) {
                auto base = edge->getBaseEdge(LOOK_BOTH | LOOK_NO_RECURRENT);
                if (base != edge && base != edges_for_same_port[0]) return base;
            }
        }
    }
    return edges_for_same_port[0];
}

bool Edge::inPlace(LOOK look) const {
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        IE_THROW() << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";
    int inputNum = getInputNum();
    int outputNum = getOutputNum();
    if (inputNum >= parentSPD->getConfig().outConfs.size())
        inputNum = 0;
    if (outputNum >= childSPD->getConfig().inConfs.size())
        outputNum = 0;

    if (look & LOOK_UP) {
        if (parentSPD->getConfig().outConfs[inputNum].inPlace() >= 0)
            return true;
    }
    if (look & LOOK_DOWN) {
        if (childSPD->getConfig().inConfs[outputNum].inPlace() >= 0)
            return true;
    }
    return false;
}

}   // namespace intel_cpu
}   // namespace ov
