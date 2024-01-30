// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"
#include "jit_kernel_emitter.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
size_t jit_brgemm_emitter::get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Input shape is original, so we need to correctly read this data by order
    // Example:
    //      Original shape (shape) = [1, 49, 2, 23]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Transposed shape = [2, 1, 49, 23]
    //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
    OV_CPU_JIT_EMITTER_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                              "detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout[layout.size() - 2];  // `1` in example
    return std::accumulate(shape.cbegin() + idx + 1, shape.end(), 1, std::multiplies<size_t>());
}
size_t jit_brgemm_emitter::get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Output shape is already transposed, we need to correctly write the data with original shape by the order
    // Example:
    //      Original transposed shape (shape) = [49, 2, 7, 39]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
    //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
    //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
    OV_CPU_JIT_EMITTER_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                              "detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout.size() - 2; // 2 in the example
    const auto dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), idx)); // 0 in the example: shape[0] = 49
    return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>()); // shape[1] x shape[2] x shape[3] = 2 x 7 x 39
}

std::atomic<int> jit_brgemm_emitter::m_next_id(0);

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr) : jit_emitter(h, isa),
    m_brgemm_id(m_next_id.fetch_add(1)) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(!brgemm_node->is_dynamic(), "Snippets don't support code generation for dynamic Brgemm");

    std::vector<size_t> leading_dimensions;
     auto get_layout = [](const std::vector<size_t>& layout, const snippets::VectorDims& io_shape) {
        if (!layout.empty()) return layout;
        std::vector<size_t> default_layout(io_shape.size());
        std::iota(default_layout.begin(), default_layout.end(), 0);
        return default_layout;
    };

    auto init_in_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& input) {
        const auto& layout = get_layout(input->get_layout(), input->get_shape());
        leading_dimensions.push_back(get_in_leading_dim(input->get_shape(), layout));
    };
    auto init_out_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& output) {
        const auto& layout = get_layout(output->get_layout(), output->get_shape());
        leading_dimensions.push_back(get_out_leading_dim(output->get_shape(), layout));
    };

    const auto& input_0_desc = expr->get_input_port_descriptor(0);
    const auto& input_1_desc = expr->get_input_port_descriptor(1);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    init_in_scheduling_params(input_0_desc);
    if (brgemm_node->is_with_data_repacking()) {
        const auto& brgemm_copy = brgemm_node->get_brgemm_copy();
        const auto& allocated_shape = brgemm_copy->get_data_repacking_shape(input_1_desc->get_shape());
        leading_dimensions.push_back(*allocated_shape.rbegin());
    } else {
        init_in_scheduling_params(input_1_desc);
    }
    init_out_scheduling_params(output_desc);

    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    const auto& outPrc = brgemm_node->get_output_element_type(0);
    bool with_amx = brgemm_node->is_amx();

    m_with_comp = brgemm_node->is_with_compensations();
    m_with_scratch = brgemm_node->is_with_scratchpad();

    const auto& output_subtensor = output_desc->get_subtensor();
    const auto& input_0_subtensor = input_0_desc->get_subtensor();
    const auto& input_1_subtensor = input_1_desc->get_subtensor();

    OV_CPU_JIT_EMITTER_ASSERT(*(output_subtensor.rbegin() + 1) == *(input_0_subtensor.rbegin() + 1),
                              "Brgemm has different M dimension subtensors on input0 and output");
    OV_CPU_JIT_EMITTER_ASSERT(*output_subtensor.rbegin() == *input_1_subtensor.rbegin(),
                              "Brgemm has different N dimension subtensors on input1 and output");
    OV_CPU_JIT_EMITTER_ASSERT(*input_0_subtensor.rbegin() == *(input_1_subtensor.rbegin() + 1),
                              "Brgemm has different K dimension subtensors on input0 and input1");

    m_ctx.M = *(output_subtensor.rbegin() + 1);
    m_ctx.N = *output_subtensor.rbegin();
    m_ctx.K = *input_0_subtensor.rbegin();
    m_ctx.LDA = leading_dimensions[0];
    m_ctx.LDB = leading_dimensions[1];
    m_ctx.LDC = leading_dimensions[2];
    m_ctx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
    m_ctx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg1Prc));
    m_ctx.dt_out = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(outPrc));
    m_ctx.beta = brgemm_node->get_beta();
    m_ctx.id = m_brgemm_id;

    std::cout << "[ Compilation ] Brgemm Type = ";
    auto type = brgemm_node->get_type();
    if (type == BrgemmCPU::Type::AMX) {
        std::cout << "AMX\n";
        std::cout << "\tid: " << m_brgemm_id << std::endl;
        std::cout << "\tM = " << m_ctx.M << std::endl;
        std::cout << "\tK = " << m_ctx.K << std::endl;
        std::cout << "\tN = " << m_ctx.N << std::endl;
    } else if (type == BrgemmCPU::Type::Floating) {
        std::cout << "Floating\n";
    } else if (type == BrgemmCPU::Type::WithCompensations) {
        std::cout << "WithCompensations\n";
    } else if (type == BrgemmCPU::Type::WithDataRepacking) {
        std::cout << "WithDataRepacking\n";
    } else {
        OPENVINO_THROW("Unknown brgemm type");
    }

    init_brgemm_kernel(m_ctx, m_kernel, with_amx);

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
    }
}

void jit_brgemm_emitter::init_brgemm_kernel(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& kernel, bool use_amx) {
    brgemm_t desc;
    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_undef
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&desc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
                                   false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, nullptr);
    if (status != dnnl_success)
        OV_CPU_JIT_EMITTER_THROW("cannot initialize brgemm descriptor due to invalid params");

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(desc, ctx.palette);

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* kernel_ = nullptr;
    status = brgemm_kernel_create(&kernel_, desc);
    if (status != dnnl_success)
        OV_CPU_JIT_EMITTER_THROW("cannot create brgemm kernel due to invalid params");
    kernel.reset(kernel_);
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT((m_with_scratch && in.size() == 3) || (!m_with_scratch && in.size() == 2),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(m_with_scratch ? in[2] : 0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        emit_brgemm_kernel_call(m_kernel.get(),
                                input_0,
                                input_1,
                                input_2,
                                output_0,
                                m_load_offset_a,
                                m_load_offset_b,
                                m_load_offset_scratch,
                                m_store_offset_c);
    } else {
        OV_CPU_JIT_EMITTER_THROW("requires at least avx512_core instruction set");
    }
}

void jit_brgemm_emitter::emit_brgemm_kernel_call(const brgemm_kernel_t *brg_kernel,
                                                 Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                                 const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                                 const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    if (m_ctx.is_with_amx) {
        internal_call_preamble();
        const auto& tile_configure_overload =
            static_cast<void (*)(jit_snippets_call_args*, brgemmCtx*)>(amx_tile_configure_if_needed);
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(tile_configure_overload));
        h->mov(abi_param2, reinterpret_cast<uintptr_t>(&m_ctx));

        internal_call_rsp_align();
        h->call(h->rbp);
        internal_call_rsp_restore();

        internal_call_postamble();
    }

    internal_call_preamble();
    // save function address in gpr to pass in call instruction
    const auto& brgemm_kernel_overload = static_cast<void (*)(const brgemm_kernel_t*,
                                                              const void*,
                                                              const void*,
                                                              void*,
                                                              void*,
                                                              int)>(kernel_execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(brgemm_kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);
    if (m_with_scratch)
        h->uni_vmovq(Xmm(3), scratch);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(brg_kernel));
    data_ptr_reg(Xmm(0), abi_param2, in0_kernel_offset);
    data_ptr_reg(Xmm(1), abi_param3, in1_kernel_offset);
    data_ptr_reg(Xmm(2), abi_param4, out0_kernel_offset);

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function brgemm_kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);
    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);

    // Push the remaining parameters on the stack
    if (m_with_scratch) {
        h->uni_vmovq(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], Xmm(3));
        if (in2_kernel_offset) h->add(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], in2_kernel_offset);
    } else {
        h->mov(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_not_param1, static_cast<int>(m_with_comp));
    h->mov(h->qword[h->rsp + (abi_param_count + 1) * gpr_size], abi_not_param1);
#else
    if (m_with_scratch) {
        data_ptr_reg(Xmm(3), abi_param5, in2_kernel_offset);
    } else {
        h->mov(abi_param5, reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_param6, static_cast<int>(m_with_comp));
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
    h->add(h->rsp, num_args_passed_on_stack * gpr_size);
#endif

    internal_call_postamble();
}

void jit_brgemm_emitter::amx_tile_configure_if_needed(jit_snippets_call_args* call_args, brgemmCtx* ctx) {
    OPENVINO_ASSERT(call_args);
    auto& amx_tile_config = call_args->tile_config;
    OPENVINO_ASSERT(amx_tile_config);
    if (ctx->M != amx_tile_config->M || ctx->K != amx_tile_config->K || ctx->N != amx_tile_config->N) {
        amx_tile_config->M = ctx->M;
        amx_tile_config->K = ctx->K;
        amx_tile_config->N = ctx->N;
        amx_tile_configure(ctx->palette);
        std::cout << " Configure\n";
    } else {
        std::cout << " Skipped\n";
    }
}

void jit_brgemm_emitter::kernel_execute(const brgemm_kernel_t *brg_kernel, const void *A, const void *B, void *C, void *scratch, int with_comp) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = A;
    brgemm_p.ptr_B = B;
    brgemm_p.ptr_C = C;
    brgemm_p.ptr_D = C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(with_comp);
    brgemm_p.do_apply_comp = static_cast<size_t>(with_comp);
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel != nullptr, "has nullptr kernel");

    (*brg_kernel)(&brgemm_p);
}

}   // namespace intel_cpu
}   // namespace ov
