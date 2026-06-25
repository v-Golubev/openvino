// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
#include "grouped_matmul_batched_gemm_gen.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/grouped_matmul.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
// clang-format on

namespace ov::intel_gpu::ocl {

// Internal buffer indices — must match GroupedMatmulBufferIdx in grouped_matmul.cpp.
static constexpr size_t GM_BUF_GROUP_EXPERT_IDS = 0;
static constexpr size_t GM_BUF_GROUP_OFFSETS    = 1;
static constexpr size_t GM_BUF_GROUP_SIZES      = 2;
static constexpr size_t GM_BUF_NUM_GROUPS       = 3;

JitConstants GroupedMatmulBatchedGemmGenerator::build_jit_constants(const kernel_impl_params& params,
                                                                     const micro::Package& bgm_gemm,
                                                                     const gathermatmul_config& cfg) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_expert_subgroup_size(device_info.arch));

    const size_t weight_idx = grouped_matmul::InputIdx::WEIGHT;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();

    add_expert_weight_quant_jit(jit, params, cfg, weight_idx);

    // Layout JIT constants.
    const auto& in_offsets_map  = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    jit.add(make_layout_jit_constants("INPUT0", params.input_layouts[grouped_matmul::InputIdx::INPUT],
                                      in_offsets_map.at(grouped_matmul::InputIdx::INPUT)));
    jit.add(make_layout_jit_constants("INPUT1", params.input_layouts[grouped_matmul::InputIdx::WEIGHT],
                                      in_offsets_map.at(grouped_matmul::InputIdx::WEIGHT)));
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

    if (bgm_gemm.getSetting("slm_size") > 0)
        jit.make("USE_SLM", 1);

    add_swiglu_jit(jit, params, weight_shape[1]);

    return jit;
}

std::mutex GroupedMatmulBatchedGemmGenerator::mtx;

Arguments GroupedMatmulBatchedGemmGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    args.push_back({ArgumentDescriptor::Types::INPUT, grouped_matmul::InputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, grouped_matmul::InputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GM_BUF_GROUP_EXPERT_IDS)});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GM_BUF_GROUP_OFFSETS)});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GM_BUF_GROUP_SIZES)});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GM_BUF_NUM_GROUPS)});
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    auto cfg = GatherMatmulMicroGenerator::get_config(params);
    if (cfg.is_weight_quantized) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
        if (!cfg.is_weight_symmetric_quantized)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});
    }
    return args;
}

DispatchDataFunc GroupedMatmulBatchedGemmGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
        assert(!params.is_dynamic());

        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_k = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_k"));
        auto wg_tile_m   = gemm_p.getSetting("wg_tile_m");
        auto wg_tile_n   = gemm_p.getSetting("wg_tile_n");

        const auto& device_info  = params.get_device_info();
        const auto& weight_shape = params.get_input_layout(grouped_matmul::InputIdx::WEIGHT).get_shape();
        const auto& a_shape      = params.get_input_layout(grouped_matmul::InputIdx::INPUT).get_shape();
        const bool fused_swiglu  = params.has_fused_primitives();
        const size_t m           = fused_swiglu ? weight_shape[1] / 2 : weight_shape[1];
        const size_t k           = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
        const size_t G           = weight_shape[0];
        const size_t T_per_group = a_shape[a_shape.size() - 2];  // M (3D) or T/G estimate (2D)

        auto& wgs    = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();

        wgs.local  = {sg_per_wg_m * get_expert_subgroup_size(device_info.arch), sg_per_wg_n, sg_per_wg_k};
        wgs.global = {ceil_div(m, wg_tile_m), ceil_div(T_per_group, wg_tile_n), G};
        wgs.global[0] *= wgs.local[0];
        wgs.global[1] *= wgs.local[1];
        wgs.global[2] *= wgs.local[2];

        scalars.push_back({ScalarDescriptor::Types::INT32, {.s32 = static_cast<int32_t>(m)}});
        scalars.push_back({ScalarDescriptor::Types::INT32, {.s32 = static_cast<int32_t>(k)}});
    }};
}

}  // namespace ov::intel_gpu::ocl
#endif
