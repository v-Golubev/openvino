// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>

namespace MKLDNNPlugin {

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                          const std::vector<InferenceEngine::TensorDesc> &outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    void createPrimitive() override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool created() const override;
    int getMaxBatch() override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() const override;

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights) const;

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;

    std::array<MKLDNNMemoryDesc, 2> in_data_d;
    MKLDNNMemoryDesc out_data_d;
};

}  // namespace MKLDNNPlugin

