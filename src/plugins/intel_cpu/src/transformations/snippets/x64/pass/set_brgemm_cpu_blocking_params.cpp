// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "set_brgemm_cpu_blocking_params.hpp"

#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
pass::SetBrgemmCPUBlockingParams::SetBrgemmCPUBlockingParams() {
    MATCHER_SCOPE(SetBrgemmCPUBlockingParams);

    auto m_brgemm = ov::pass::pattern::wrap_type<BrgemmCPU>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::SetBrgemmCPUBlockingParams")
        const auto node = m.get_match_root();
        auto brgemm = ov::as_type_ptr<BrgemmCPU>(node);

        const auto& input_1_precision = brgemm->get_input_element_type(1);
        // Ticket: 113745
        // TODO: extend block size selection heuristics
        auto get_block_size_m = [&](const size_t M) -> size_t {
            return 32;
        };
        auto get_block_size_k = [&](const size_t K) -> size_t {
            // K blocking is disabled in dynamism by default
            if (ov::snippets::utils::is_dynamic_value(K))
                return snippets::utils::get_dynamic_value<size_t>();

            if (input_1_precision != ov::element::f32)
                return K;
            return K > 1024 ? 1024 : K > 512 ? 512 : K;
        };
        auto get_block_size_n = [&](const size_t N) -> size_t {
            // N blocking is disabled in dynamism by default
            if (ov::snippets::utils::is_dynamic_value(N))
                return snippets::utils::get_dynamic_value<size_t>();
            return input_1_precision == ov::element::f32 ? 64 : N;
        };

        const auto brgemm_in0_dims = snippets::utils::get_planar_pshape(brgemm->input(0));
        const auto brgemm_in1_dims = snippets::utils::get_planar_pshape(brgemm->input(1));
        const auto M = ov::snippets::utils::dimension_to_size_t(*++brgemm_in0_dims.rbegin());
        const auto K = ov::snippets::utils::dimension_to_size_t(*brgemm_in0_dims.rbegin());
        const auto N = ov::snippets::utils::dimension_to_size_t(*brgemm_in1_dims.rbegin());
        const auto m_blk = get_block_size_m(M);
        const auto k_blk = get_block_size_k(K);
        const auto n_blk = get_block_size_n(N);

        if (brgemm->is_with_data_repacking()) {
            const auto brgemm_copy_b = brgemm->get_brgemm_copy();
            const auto& copy_b_in_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(brgemm_copy_b->input(0));
            const auto& input_layout = copy_b_in_desc->get_layout();
            const bool transpose_b_case = !input_layout.empty() && input_layout[input_layout.size() - 1] != input_layout.size() - 1;
            // Note: transpose repacking is not covered in blocking loops in order to limit potentially negative performance impact
            // TODO: enable blocking for repacking with transpose
            if (transpose_b_case) {
                brgemm_copy_b->set_k_block_size(K);
                brgemm_copy_b->set_n_block_size(N);
            } else {
                const auto brgemmVNNIFactor = brgemm_copy_b->get_brgemm_vnni_factor();
                OPENVINO_ASSERT(ov::snippets::utils::is_dynamic_value(K) || k_blk == K || k_blk % brgemmVNNIFactor == 0,
                                "K Block size (",
                                k_blk,
                                "), which is not divisible by brgemmVNNIFactor (",
                                brgemmVNNIFactor,
                                ") and not equal to K dimension (",
                                K,
                                "), is not supported for brgemm data repacking.");
                brgemm_copy_b->set_k_block_size(k_blk);
                brgemm_copy_b->set_n_block_size(n_blk);
            }
        }

        // brgemm->set_m_block_size(m_blk);
        // brgemm->set_k_block_size(k_blk);
        // brgemm->set_n_block_size(n_blk);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov