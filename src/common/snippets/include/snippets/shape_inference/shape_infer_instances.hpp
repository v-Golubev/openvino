// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference.hpp"

namespace ov {
namespace snippets {
class NumpyBroadcastShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};


template<class BroadcastOP>
class BroadcastShapeInfer : public IShapeInferSnippets {
    VectorDims::value_type m_broadcasted_dim;
public:
    explicit BroadcastShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    VectorDims::value_type get_broadcasted_dim() const;
    void set_broadcasted_dim(const VectorDims::value_type new_dim);
};

class PassThroughShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(!input_shapes.empty(), "Empty Input shapes are not allowed for PassThroughShapeInfer");
        return {{input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

class EmptyShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{}, ShapeInferStatus::success};
    }
};

class SingleElementShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{{1}}, ShapeInferStatus::success};
    }
};

class SelectShapeInfer : public IShapeInferSnippets {
    ov::op::AutoBroadcastSpec m_broadcast_spec;
public:
    explicit SelectShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class HorizonOpShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

} // namespace snippets
} // namespace ov
