// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void enable_keep_original_precision(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void disable_keep_original_precision(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_keep_original_precision(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief KeepOriginalPrecision class represents runtime info attribute that marks nodes
 * as prohibitted to fuse precision in ConvertPrecision
 */
class TRANSFORMATIONS_API KeepOriginalPrecision : public RuntimeAttribute {
public:
    OPENVINO_RTTI("keep_original_precision", "0");
    KeepOriginalPrecision() = default;
};

}  // namespace ov
