// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_3gemm_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    using namespace ov::pass::pattern;

    auto moe_compressed_m = wrap_type<ov::intel_gpu::op::MOECompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(pattern_map.at(moe_compressed_m).get_node_shared_ptr());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }

        auto config = moe_compressed->get_config();

        // Pass all inputs through: hidden_states, routing_weights, topk_indices, w0-w2 weights/scales/zps
        OutputVector args;
        for (size_t i = 0; i < moe_compressed->get_input_size(); ++i) {
            args.push_back(moe_compressed->input_value(i));
        }

        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);
        moe_3gemm_fused_compressed->set_friendly_name(moe_compressed->get_friendly_name());
        ov::copy_runtime_info(moe_compressed, moe_3gemm_fused_compressed);
        ov::replace_node(moe_compressed, moe_3gemm_fused_compressed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
