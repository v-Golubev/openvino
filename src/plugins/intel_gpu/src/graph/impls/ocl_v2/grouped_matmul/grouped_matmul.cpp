// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "grouped_matmul_batched_gemm_gen.hpp"
// clang-format on

#include "../expert_gemm_gen_utils.hpp"
#include "../primitive_ocl_base.hpp"
#include "grouped_matmul_impl.hpp"
#include "grouped_matmul_inst.h"
#include "intel_gpu/primitives/grouped_matmul.hpp"
#include "intel_gpu/primitives/swiglu.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <unordered_map>

#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

namespace ov::intel_gpu::ocl {
namespace {

static bool weights_need_ocl_batched(const kernel_impl_params& params) {
    const auto& wl = params.input_layouts[grouped_matmul::InputIdx::WEIGHT];
    return (wl.data_type == cldnn::data_types::u4 || wl.data_type == cldnn::data_types::i4);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
inline dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    return to_onednn_dtype(dt);
}
#endif

// Internal buffer layout for the OCL batched path.
// These 4 buffers replace the 9 buffers of gather_matmul (no sort/scatter needed).
enum GroupedMatmulBufferIdx {
    GROUP_EXPERT_IDS = 0,  // i32[G]  — identity: buf[g] = g
    GROUP_OFFSETS    = 1,  // i32[G]  — cumulative start row for group g
    GROUP_SIZES      = 2,  // i32[G]  — number of tokens in group g
    NUM_GROUPS_BUF   = 3,  // i32[1]  — G
};

class GroupedMatmulOCLImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupedMatmulOCLImpl)

#ifdef ENABLE_ONEDNN_FOR_GPU
    Stage::Ptr ocl_batched = make_stage<GroupedMatmulBatchedGemmGenerator>();
#endif

    explicit GroupedMatmulOCLImpl() : PrimitiveImplOCL(GroupedMatmulImpl::get_type_info_static()) {}
    explicit GroupedMatmulOCLImpl(const RuntimeParams& impl_param) : GroupedMatmulOCLImpl() {
#ifdef ENABLE_ONEDNN_FOR_GPU
        auto params = impl_param;
        if (weights_need_ocl_batched(params)) {
            add_stage(ocl_batched, params);
        }
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupedMatmulOCLImpl>(this);
    }

    // init_kernels: only the OCL path needs kernels; skip for the oneDNN path.
    void init_kernels(const cldnn::kernels_cache& kernels_cache, const RuntimeParams& params) override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (weights_need_ocl_batched(params)) {
            PrimitiveImplOCL::init_kernels(kernels_cache, params);
        }
        // oneDNN path: no OCL kernels needed.
#endif
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (!weights_need_ocl_batched(params))
            return {};  // oneDNN path: no internal buffers

        const auto& b_shape = params.input_layouts[grouped_matmul::InputIdx::WEIGHT].get_shape();
        const size_t G = b_shape[0];

        return {
            {G, ov::element::i32},  // GROUP_EXPERT_IDS
            {G, ov::element::i32},  // GROUP_OFFSETS
            {G, ov::element::i32},  // GROUP_SIZES
            {1, ov::element::i32},  // NUM_GROUPS
        };
#else
        return {};
#endif
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        inst.update_shape_info_tensor(impl_params);
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (weights_need_ocl_batched(impl_params))
            fill_group_metadata(inst, impl_params);
#endif
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        const auto& impl_params = *instance.get_impl_params();
        if (weights_need_ocl_batched(impl_params) && has_stage(ocl_batched)) {
            return execute_stage(events, instance, ocl_batched);
        }

        // oneDNN fallback.
        for (const auto& e : events)
            if (e) e->wait();

        auto desc = impl_params.typed_desc<grouped_matmul>();
        if (!desc->has_offsets)
            return execute_onednn_3d(instance);
        else
            return execute_onednn_2d(instance);
#else
        OPENVINO_THROW("grouped_matmul is only supported on systolic platforms with oneDNN.");
        return nullptr;
#endif
    }

private:
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Fill GROUP_EXPERT_IDS / GROUP_OFFSETS / GROUP_SIZES / NUM_GROUPS from
    // the current input shapes (or from the offsets input for 2D×3D).
    void fill_group_metadata(primitive_inst& inst, const kernel_impl_params& impl_params) {
        auto desc = impl_params.typed_desc<grouped_matmul>();
        const auto& b_shape = impl_params.input_layouts[grouped_matmul::InputIdx::WEIGHT].get_shape();
        const size_t G = b_shape[0];
        auto& intermediates = inst.get_intermediates_memories();
        if (intermediates.size() <= NUM_GROUPS_BUF) return;

        auto& stream = inst.get_network().get_stream();

        // Write identity expert ids and num_groups once.
        {
            std::vector<int32_t> expert_ids(G), num_groups(1, static_cast<int32_t>(G));
            for (size_t g = 0; g < G; ++g) expert_ids[g] = static_cast<int32_t>(g);
            intermediates[GROUP_EXPERT_IDS]->copy_from(stream, expert_ids.data(), 0, 0, G * sizeof(int32_t), true);
            intermediates[NUM_GROUPS_BUF]->copy_from(stream, num_groups.data(), 0, 0, sizeof(int32_t), true);
        }

        // Offsets and sizes.
        std::vector<int32_t> offsets(G), sizes(G);
        if (!desc->has_offsets) {
            // 3D×3D: uniform M per group.
            const auto& a_shape = impl_params.input_layouts[grouped_matmul::InputIdx::INPUT].get_shape();
            const int32_t M = static_cast<int32_t>(a_shape[a_shape.size() - 2]);
            for (size_t g = 0; g < G; ++g) {
                offsets[g] = static_cast<int32_t>(g) * M;
                sizes[g] = M;
            }
        } else {
            // 2D×3D: read cumulative end-offsets from input[2].
            const auto& offsets_mem = *inst.input_memory_ptr(grouped_matmul::InputIdx::OFFSETS);
            std::vector<int32_t> cum_ends(G);
            offsets_mem.copy_to(stream, cum_ends.data(), 0, 0, G * sizeof(int32_t), true);
            int32_t prev = 0;
            for (size_t g = 0; g < G; ++g) {
                offsets[g] = prev;
                sizes[g] = cum_ends[g] - prev;
                prev = cum_ends[g];
            }
        }
        intermediates[GROUP_OFFSETS]->copy_from(stream, offsets.data(), 0, 0, G * sizeof(int32_t), true);
        intermediates[GROUP_SIZES]->copy_from(stream, sizes.data(), 0, 0, G * sizeof(int32_t), true);
    }

    // ---- oneDNN fallback paths (f16 B weights) ----

    struct DnnlKernel {
        dnnl::matmul::primitive_desc pd;
        dnnl::matmul prim;
    };
    cldnn::LruCache<int64_t, std::shared_ptr<DnnlKernel>> _kernel_cache{32};

    event::ptr execute_onednn_3d(primitive_inst& instance) {
        const auto& impl_params = *instance.get_impl_params();
        const auto& input_layout  = impl_params.input_layouts[grouped_matmul::InputIdx::INPUT];
        const auto& weight_layout = impl_params.input_layouts[grouped_matmul::InputIdx::WEIGHT];
        const auto& output_layout = impl_params.output_layouts[0];

        const auto& a_shape = input_layout.get_shape();
        const auto& b_shape = weight_layout.get_shape();
        OPENVINO_ASSERT(a_shape.size() >= 3 && b_shape.size() >= 2);

        const dnnl::memory::dim G = a_shape[a_shape.size() - 3];
        const dnnl::memory::dim M = a_shape[a_shape.size() - 2];
        const dnnl::memory::dim K = a_shape[a_shape.size() - 1];
        const dnnl::memory::dim N = b_shape[b_shape.size() - 2];
        const int64_t cache_key = G * 1000003LL + M * 100003LL + N * 1009LL + K;

        if (!_kernel_cache.has(cache_key)) {
            auto& dnnl_engine = instance.get_network().get_engine().get_onednn_engine();
            dnnl::primitive_attr attr;
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto src_md = dnnl::memory::desc({G, M, K}, convert_data_type(input_layout.data_type), dnnl::memory::format_tag::abc);
            auto w_md   = dnnl::memory::desc({G, K, N}, convert_data_type(weight_layout.data_type), dnnl::memory::format_tag::acb);
            auto dst_md = dnnl::memory::desc({G, M, N}, convert_data_type(output_layout.data_type), dnnl::memory::format_tag::abc);
            auto gk = std::make_shared<DnnlKernel>();
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, dst_md, attr);
            gk->prim = dnnl::matmul(gk->pd);
            _kernel_cache.add(cache_key, gk);
        }

        auto& gk = *_kernel_cache.get(cache_key);
        auto& stream = instance.get_network().get_stream();
        auto& dnn_stream = stream.get_onednn_stream();
        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC,     instance.input_memory_ptr(grouped_matmul::InputIdx::INPUT)->get_onednn_memory(gk.pd.src_desc())},
            {DNNL_ARG_WEIGHTS, instance.input_memory_ptr(grouped_matmul::InputIdx::WEIGHT)->get_onednn_memory(gk.pd.weights_desc())},
            {DNNL_ARG_DST,     instance.output_memory_ptr(0)->get_onednn_memory(gk.pd.dst_desc())},
        };
        gk.prim.execute(dnn_stream, args);
        dnn_stream.wait();
        return stream.create_user_event(true);
    }

    event::ptr execute_onednn_2d(primitive_inst& instance) {
        const auto& impl_params = *instance.get_impl_params();
        const auto& input_layout  = impl_params.input_layouts[grouped_matmul::InputIdx::INPUT];
        const auto& weight_layout = impl_params.input_layouts[grouped_matmul::InputIdx::WEIGHT];
        const auto& output_layout = impl_params.output_layouts[0];

        const auto& a_shape = input_layout.get_shape();
        const auto& b_shape = weight_layout.get_shape();
        OPENVINO_ASSERT(a_shape.size() >= 2 && b_shape.size() >= 3);

        const dnnl::memory::dim T = a_shape[a_shape.size() - 2];
        const dnnl::memory::dim K = a_shape[a_shape.size() - 1];
        const dnnl::memory::dim G = b_shape[b_shape.size() - 3];
        const dnnl::memory::dim N = b_shape[b_shape.size() - 2];
        const int64_t cache_key = T * 1000003LL + K * 100003LL + G * 1009LL + N;

        if (!_kernel_cache.has(cache_key)) {
            auto& dnnl_engine = instance.get_network().get_engine().get_onednn_engine();
            dnnl::primitive_attr attr;
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto src_md = dnnl::memory::desc::grouped({T, K}, convert_data_type(input_layout.data_type), 0, G, dnnl::memory::data_type::s32);
            auto dst_md = dnnl::memory::desc::grouped({T, N}, convert_data_type(output_layout.data_type), 0, G, dnnl::memory::data_type::s32);
            auto w_md   = dnnl::memory::desc({G, K, N}, convert_data_type(weight_layout.data_type), dnnl::memory::format_tag::acb);
            auto gk = std::make_shared<DnnlKernel>();
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, dst_md, attr);
            gk->prim = dnnl::matmul(gk->pd);
            _kernel_cache.add(cache_key, gk);
        }

        auto& gk = *_kernel_cache.get(cache_key);
        auto& stream = instance.get_network().get_stream();
        auto& dnn_stream = stream.get_onednn_stream();
        auto& offsets_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::OFFSETS);
        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC,     instance.input_memory_ptr(grouped_matmul::InputIdx::INPUT)->get_onednn_grouped_memory(gk.pd.src_desc(), offsets_mem)},
            {DNNL_ARG_WEIGHTS, instance.input_memory_ptr(grouped_matmul::InputIdx::WEIGHT)->get_onednn_memory(gk.pd.weights_desc())},
            {DNNL_ARG_DST,     instance.output_memory_ptr(0)->get_onednn_grouped_memory(gk.pd.dst_desc(), offsets_mem)},
        };
        gk.prim.execute(dnn_stream, args);
        dnn_stream.wait();
        return stream.create_user_event(true);
    }
#endif
};
}  // namespace

std::unique_ptr<primitive_impl> GroupedMatmulImpl::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<grouped_matmul>());
    return std::make_unique<GroupedMatmulOCLImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::grouped_matmul)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupedMatmulOCLImpl)
