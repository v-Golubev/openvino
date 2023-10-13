// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/keep_original_precision.hpp"

void ov::enable_keep_original_precision(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[KeepOriginalPrecision::get_type_info_static()] = KeepOriginalPrecision{};
}

void ov::disable_keep_original_precision(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(KeepOriginalPrecision::get_type_info_static());
}

bool ov::is_keep_original_precision(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(KeepOriginalPrecision::get_type_info_static());
}
