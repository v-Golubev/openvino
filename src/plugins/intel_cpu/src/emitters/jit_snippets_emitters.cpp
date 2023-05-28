// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "jit_snippets_emitters.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"
#include "snippets_transformations/op/brgemm_copy_b.hpp"
#include "snippets_transformations/op/brgemm_cpu.hpp"

using namespace InferenceEngine;
using ngraph::snippets::op::Subgraph;
using ngraph::snippets::AllocatedEmitter;
using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

namespace {
constexpr size_t gpr_size = 8;
} // namespace

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
}

jit_container_emitter::jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                      const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                            std::vector<AllocatedEmitter>& allocated_emitters) const {
    if (allocated_emitters.empty())
        IE_THROW() << "Cannot map registers when there is no allocated_emitters provided";
    auto map_regs = [](const std::vector<size_t>& abstract_regs, mapping_info& mapping) {
        auto& abstract_to_physical = mapping.first;
        auto& regs_pool = mapping.second;
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++) {
            const auto abstract = abstract_regs[i];
            auto& physical = physical_regs[i];
            if (abstract_to_physical.count(abstract) == 0) {
                if (regs_pool.empty())
                    IE_THROW() << "Cannot map registers for jit_container_emitter: not enough regs in the pool";
                physical = regs_pool.back();
                regs_pool.pop_back();
                abstract_to_physical[abstract] = physical;
            } else {
                physical = abstract_to_physical[abstract];
            }
        }
        return physical_regs;
    };

    for (auto& code : allocated_emitters) {
        const auto& emitter = code.first;
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = code.second;
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                // Note that gpr_to_gpr is used for high-level utility operations like Kernel/Loop.
                // Input registers are not mapped in this case, since they contain utility info
                // (num_params, loop increment, etc.), but not reg indexes.
                // todo: Note that LoopBeginEmitter and LoopEndEmitter demonstrate new paradigm,
                //  where all utility emitters align with conventional Op emitters
                if (std::dynamic_pointer_cast<LoopBeginEmitter>(emitter) ||
                    std::dynamic_pointer_cast<LoopEndEmitter>(emitter) ||
                    std::dynamic_pointer_cast<BrgemmEmitter>(emitter) ||
                    std::dynamic_pointer_cast<BrgemmCopyBEmitter>(emitter))
                    in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                else
                    in_physical_regs = std::move(in_abstract_regs);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            default:
                IE_THROW() << "Unhandled in_out type";
        }
        code.second = std::make_pair(in_physical_regs, out_physical_regs);
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(code.first))
            container->map_abstract_registers(gpr_map_pool,  vec_map_pool, allocated_emitters);
    }
}

KernelEmitter::KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) :
    jit_container_emitter(h, isa, n),
    reg_indexes_idx(abi_param1.getIdx()),
    reg_const_params_idx(abi_param2.getIdx()) {
    const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
    if (!kernel)
        IE_THROW() << "KernelEmitter invoked with invalid op argument";
    if (kernel->region.empty())
        IE_THROW() << "KernelEmitter invoked with empty body";
    if (kernel->compile_params == nullptr)
        IE_THROW() << "KernelEmitter invoked with op::Kernel that contains no compile_params";
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    // calc data access pattern. we'll need it for offsets calculation
    const auto&  model = kernel->model;
    const auto get_data_layout = [](const Output<ov::Node>& out, std::vector<size_t>& shape) {
        auto node = out.get_node_shared_ptr();
        while (ov::is_type<ngraph::snippets::op::LoopEnd>(node)) {
            node = node->get_input_node_shared_ptr(out.get_index());;
        }
        const auto& layout = ngraph::snippets::utils::get_node_output_layout(node);
        // default access pattern
        if (!layout.empty()) {
            const auto layout_shape_diff = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(layout.size());
            // Plugin can (and usually does) prepend shapes with 1's to facilitate scheduling, here we can safely remove leading 1's
            if (layout_shape_diff > 0) {
                if (std::any_of(shape.begin(), shape.begin() + layout_shape_diff, [](size_t x){return x != 1;}))
                    IE_THROW() << "KernelEmitter detected shape vs access pattern conflict: only leading 1's can be removed from the shape";
                shape.erase(shape.begin(), shape.begin() + layout_shape_diff);
            }
        }
        return layout;
    };
    const auto& ops = model->get_ordered_ops();
    auto params = model->get_parameters();
    auto results = model->get_results();
    num_inputs = params.size();
    num_outputs = results.size();
    std::set<size_t> unique_buffers;
    for (const auto& op : ops) {
        if (const auto buffer = ov::as_type_ptr<ngraph::snippets::op::Buffer>(op))
            unique_buffers.insert(buffer->get_id());
    }
    num_unique_buffers = unique_buffers.size();
    NodeVector io_nodes;
    std::copy(params.begin(), params.end(), std::back_inserter(io_nodes));
    std::copy(results.begin(), results.end(), std::back_inserter(io_nodes));

    const auto& model_rt_info = model->get_rt_info();
    const auto& plugin_shapes = model_rt_info.find("PluginShapesOverride");
    if (plugin_shapes == model_rt_info.end()) {
        IE_THROW() << "JIT KernelEmitter requires plugin-overriden shapes in model rt_info";
    } else {
        const auto& new_shapes = plugin_shapes->second.as<std::vector<std::vector<size_t>>>();
        if (new_shapes.size() != num_inputs + num_outputs)
            IE_THROW() << "JIT KernelEmitter detected invalid plugin-overriden shapes";
        io_shapes = new_shapes;
    }
    for (int i = 0; i < io_nodes.size(); i++) {
        const auto& out = i < num_inputs ? io_nodes[i]->output(0) : io_nodes[i]->input_value(0);
        data_layout.push_back(get_data_layout(out, io_shapes[i]));
        io_data_size.push_back(out.get_element_type().size());
    }
    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    // todo: it's more convenient to use std::set as a pool container (unique and always sorted),
    //  but pools are vectors to align with emit_code signature. Change signature?
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         reg_indexes_idx, reg_const_params_idx});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    std::vector<AllocatedEmitter> data_io_emitters;
    std::copy_if(body.begin(), body.end(), std::back_inserter(data_io_emitters),
                           [](const AllocatedEmitter& code){
                                   const auto& emitter = code.first;
                                   const auto emitter_type = std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type();
                                   // todo: how this will be handled if Brgemm in & out are op::Buffer
                                   // Brgemm is a special case since it incorporates input and output (we use onednn kernel)
                                   // Just like Load & Store it requires offsets calculation
                                   const auto is_brgemm = std::dynamic_pointer_cast<BrgemmEmitter>(emitter) ||
                                                          std::dynamic_pointer_cast<BrgemmCopyBEmitter>(emitter);
                                   return emitter_type == gpr_to_vec || emitter_type == vec_to_gpr || is_brgemm;
                           });
    // Note that we can't use reg_indexes_idx or reg_const_params_idx to store data pointers because these two
    // regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, data_io_emitters);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    // However we can use reg_indexes_idx and reg_const_params_idx for other operations since we won't need them
    // after offsets calculation
    gpr_map_pool.second.push_back(reg_indexes_idx);
    gpr_map_pool.second.push_back(reg_const_params_idx);
    map_abstract_registers(gpr_map_pool, vec_map_pool, body);
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out) const {
    if (!in.empty())
        IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 0, got " << in.size();
    if (!out.empty())
        IE_THROW() << "KernelEmitter got invalid number of outputs. Expected 0, got " << out.size();
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    if (data_ptr_regs_idx.size() != num_params)
        IE_THROW() << "KernelEmitter: number of inputs and outputs is inconsisnent with the number of allocated registers "
        << num_params << " data_ptr_regs_idx.size() = " << data_ptr_regs_idx.size();
}

void KernelEmitter::init_data_pointers(const Reg64& reg_indexes, const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = jcp.master_shape.size() - 1;
    //const size_t tile_rank = jcp.tile_rank;
    const auto num_params = num_inputs + num_outputs;
    std::vector<std::vector<size_t>> data_offsets(num_params, std::vector<size_t>{});
    auto offset_calculation = [=](const std::vector<size_t>& shape,
                                            const std::vector<size_t>& layout, const size_t data_size) {
        // Strides represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    strides: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    strides: s1*s3, s3,       0,  1
        std::vector<size_t> strides(shape.size());
        size_t dim_step = 1;
        strides[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            dim_step *= shape[k+1];
            strides[k] = shape[k] != 1 ? dim_step * data_size : 0;
        }
        // Note: this is an extra copy, but let's keep it for clarity
        if (!layout.empty()) {
            std::vector<size_t> reordered_strides(strides.size());
            for (auto i = 0; i < layout.size(); i++)
                reordered_strides[i] = strides[layout[i]];
            strides = std::move(reordered_strides);
        }
        // the last stride is ignored, since the entire last dim is processed by kernel
        // and no parallel_for data_ptr offsets can be applied in this case (cover tile_rank == 1)
        strides.pop_back();
        // if tile_rank > 1, then zero corresponding strides since no external offset can be applied
        // for (auto j = 0; j < tile_rank - 1; j++)
        //    strides[strides.size() - 1 - j] = 0;
        // actual offset size might be larger that the shape size due to 6D scheduling
        strides.insert(strides.begin(), offset_rank - strides.size(), 0);

        return strides;
    };
    for (size_t i = 0; i < num_params; i++) {
        data_offsets[i] = offset_calculation(io_shapes[i],  data_layout[i], io_data_size[i]);
    }
    // master_shape size must be valid in both static and dynamic cases
    std::function<void(Reg64, const std::vector<size_t>&, Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Reg64 pointer, const std::vector<size_t>& offsets, Reg64 reg_tmp) {
        for (int j = 0; j < offset_rank; j++) {
            if (jcp.master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_const_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : Reg64(static_cast<int>(*spare_corruptable_gpr));
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_const_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_const_params;
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    h->preamble();

    Reg64 reg_indexes = Reg64(static_cast<int>(reg_indexes_idx));
    Reg64 reg_const_params = Reg64(static_cast<int>(reg_const_params_idx));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_indexes, reg_const_params, data_ptr_regs);
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}


LoopBeginEmitter::LoopBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>(n);
    if (!loop_begin)
        IE_THROW() << "LoopBeginEmitter invoked with invalid op argument";
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        IE_THROW() << "LoopBeginEmitter invoked with invalid configuration: the last output must have exactly one input attached";
    const auto loop_end = ov::as_type_ptr<ngraph::snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        IE_THROW() << "LoopBeginEmitter invoked with invalid configuration: the last output must be LoopEnd";
    work_amount = loop_begin->get_work_amount();
    evaluate_once = loop_begin->get_evaluate_once();
    num_inputs = loop_begin->get_input_size();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void LoopBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    if (in.size() != num_inputs)
        IE_THROW() << "Invalid inputs size: expected " << num_inputs << " got " << in.size();
    if (out.size() != num_inputs + 1)
        IE_THROW() << "Invalid outputs size: expected " << num_inputs + 1 << " got " << out.size();
}

void LoopBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    // todo: In dynamic case we will also need to set broadcasting info here
    Reg64 reg_work_amount = Reg64(out.back());
    Label for_body;
    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
    loop_begin->input_regs = in;
}

LoopEndEmitter::LoopEndEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    loop_end = ov::as_type_ptr<ngraph::snippets::op::LoopEnd>(n);
    if (!loop_end)
        IE_THROW() << "LoopEndEmitter invoked with invalid op argument";
    loop_begin = loop_end->get_loop_begin();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!loop_begin)
        IE_THROW() << "LoopEndEmitter invoked with invalid configuration: the last arg must be LoopBegin";
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = loop_begin->get_input_size();
    num_outputs = loop_end->get_output_size();
    wa_increment = loop_end->get_increment();
    work_amount = loop_end->get_work_amount();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    for (int i = 0; i < num_inputs; i++)
        io_data_size.push_back(static_cast<int64_t>(loop_begin->get_input_element_type(i).size()));
    for (int i = 0; i < num_outputs; i++)
        io_data_size.push_back(static_cast<int64_t>(loop_end->get_output_element_type(i).size()));
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}


void LoopEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out) const {
    if (loop_begin->input_regs.size() != num_inputs)
        IE_THROW() << "Invalid loop_begin->input_regs size: expected " << num_inputs << " got " << loop_begin->input_regs.size();
    if (out.size() != num_outputs)
        IE_THROW() << "Invalid number of out arguments: expected " << num_outputs << " got " << out.size();
    if (in.size() != num_outputs + 1)
        IE_THROW() << "Invalid number of in arguments: expected " << num_inputs + 1 << " got " << in.size();
    const auto io_size = num_inputs + num_outputs;
    if (ptr_increments.size() != io_size)
        IE_THROW() << "Invalid apply_increments size: expected " << io_size << " got " << ptr_increments.size();
    if (finalization_offsets.size() != io_size)
        IE_THROW() << "Invalid finalization_offsets size: expected: " << io_size << " got " << finalization_offsets.size();
}

void LoopEndEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs(loop_begin->input_regs);
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(out.begin(), out.end(), std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(in.back());
    if (!evaluate_once) {
        for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], ptr_increments[idx] * io_data_size[idx]);
        }
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(loop_begin->begin_address);
    }

    for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

BroadcastMoveEmitter::BroadcastMoveEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    if (n->get_input_element_type(0) != n->get_output_element_type(0))
        IE_THROW() << "BroadcastMoveEmitter supports only equal input and output types but gets: "
            << n->get_input_element_type(0) << " and " << n->get_output_element_type(0);
    byte_size = n->get_input_element_type(0).size();
}

void BroadcastMoveEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastMove emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastMoveEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Xmm xmm_src0 = Xmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    switch (byte_size) {
        case 4: h->uni_vbroadcastss(vmm_dst, xmm_src0); break;
        case 2: h->vpbroadcastw(vmm_dst, xmm_src0); break;
        case 1: h->vpbroadcastb(vmm_dst, xmm_src0); break;
        default: assert(!"unsupported data type");
    }
}

ScalarEmitter::ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto precision = n->get_output_element_type(0);
    switch (precision) {
        case element::i32: {
            value = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<int32_t>()[0];
            break;
        }
        case element::f32: {
            value = dnnl::impl::cpu::x64::float2int(ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<float>()[0]);
            break;
        }
        default: {
            IE_THROW() << "Scalar emitter doesn't support " << precision;
        }
    }
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Scalar emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_dst  = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}


MemoryEmitter::MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    src_prc = InferenceEngine::details::convertPrecision(n->get_input_element_type(0));
    dst_prc = InferenceEngine::details::convertPrecision(n->get_output_element_type(0));
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "StoreEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(n);
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Store emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "LoadEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    const auto load = std::dynamic_pointer_cast<ngraph::snippets::op::Load>(n);
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Load emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "BroadcastEmitters support only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    const auto broadcast_load = std::dynamic_pointer_cast<ngraph::snippets::op::BroadcastLoad>(n);
    byte_offset = broadcast_load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void BroadcastLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastLoad emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        default: assert(!"unsupported data type");
    }
}

LoadConvertEmitter::LoadConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(n);
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "LoadConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadConvertEmitter::emit_data() const {
    load_emitter->emit_data();
}

StoreConvertEmitter::StoreConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(n);
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    }
}

void StoreConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "StoreConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreConvertEmitter::emit_data() const {
    store_emitter->emit_data();
}
size_t BrgemmEmitter::getBrgIdx(size_t kIdx, size_t nIdx) {
    return kIdx * BRGEMM_KERNELS_NUM[1] + nIdx;
}
BrgemmEmitter::BrgemmEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& node) : jit_emitter(h, isa, node) {
    m_brgCtxs.fill(brgemmCtx());
    std::generate(m_brgKernels.begin(), m_brgKernels.end(), [](){return nullptr;});
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    if (brgemm_node->is_dynamic())
        IE_THROW() << "Snippets don't support code generation for dynamic Brgemm";
    const auto brgemm_copy = brgemm_node->is_with_data_repacking() ? brgemm_node->get_brgemm_copy() : nullptr;
    const OutputVector io_values {brgemm_node->input_value(0),
                                  brgemm_copy ? brgemm_copy->input_value(0) : brgemm_node->input_value(1),
                                  brgemm_node->output(0)};
    // Todo: check the case of brgemm copy before the merge
    std::vector<size_t> leading_dimensions{brgemm_node->get_leading_dim_a(),
                                           brgemm_node->get_leading_dim_b(),
                                           brgemm_node->get_leading_dim_c()};
    std::vector<std::vector<size_t>> io_layouts;
    for (const auto& val : io_values) {
        const auto& layout = ngraph::snippets::utils::get_node_output_layout(val.get_node_shared_ptr());
        if (layout.empty()) {
            // empty value indicates a planar layout
            std::vector<size_t> default_layout(val.get_shape().size());
            std::iota(default_layout.begin(), default_layout.end(), 0);
            io_layouts.push_back(default_layout);
        } else {
            io_layouts.push_back(layout);
        }
    }

    const auto& A_shape = io_values[0].get_shape();
    const auto& A_layout = io_layouts[0];
    const auto& C_shape = io_values[2].get_shape();
    const auto& C_layout = io_layouts[2];

    // We need find original M,N,K having layouts and ordered shapes
    // Layout:  0, 1, 2, 3   =>   New layout: 0, 2, 1, 3
    // Shape:   1, 3, 5, 9   =>   New Shape:  1, 5, 3, 9
    // To find original 2nd dimension, we should find index of position value `2` in new layout
    // and get dimension from new shape by this index
    auto get_ordered_idx = [](const std::vector<size_t>& layout, size_t idx) {
        return std::distance(layout.begin(), std::find(layout.begin(), layout.end(), idx));
    };

    m_M = brgemm_node->get_input_count(0);
    if (m_M == 0)
        return;
    m_K = A_shape[get_ordered_idx(A_layout, A_layout.size() - 1)];
    // B_shape[B_layout[3]]
    m_N = C_shape[get_ordered_idx(C_layout, C_layout.size() - 1)];

    auto brg0Prc = InferenceEngine::details::convertPrecision(brgemm_node->get_input_element_type(0));
    auto brg1Prc = InferenceEngine::details::convertPrecision(brgemm_node->get_input_element_type(1));
    io_data_size = {brg0Prc.size(), brg1Prc.size(), brgemm_node->get_output_element_type(0).size()};
    if (brgemm_node->get_input_size() > 2)
        io_data_size.insert(io_data_size.begin() + 2, brgemm_node->get_input_element_type(2).size());
    m_brg0VnniFactor = 4 / brg0Prc.size();
    bool brgWithAMX = brgemm_node->is_amx();

    m_with_comp = brgemm_node->is_with_compensations();
    m_with_scratch = brgemm_node->is_with_scratchpad();

//    m_N_blk = brg1Prc == Precision::FP32 ? m_N :
//              brg1Prc == Precision::BF16 ? 32 : 64;
//    m_K_blk = brgWithAMX ? brg0Prc == Precision::BF16 ? 32 : 64
//                         : m_K;
    m_N_blk = 64;
    m_K_blk = 1024;
    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;
    m_N_blocking_loop_needed = m_N >= 2 * m_N_blk;
    // Note: K dimension is covered by TWO blocked kernels (with beta = 0 and 1)
    // so we need a K loop only if more than 2 blocked + tail are required
    m_K_blocking_loop_needed = m_K >= 3 * m_K_blk;
    bool has_K_kernel = false;
    for (size_t k = 0; k < BRGEMM_KERNELS_NUM[0]; k++) {
        bool has_N_kernel = false;
        for (size_t n = 0; n < BRGEMM_KERNELS_NUM[1]; n++) {
            const size_t kernel_idx = getBrgIdx(k, n);

            auto& brgemmCtx = m_brgCtxs[kernel_idx];

            brgemmCtx.M = m_M;
            brgemmCtx.N = n == 0 ? m_N_blk : m_N_tail;
            // The first two kernels are blocked in case of K
            brgemmCtx.K = [=]() {
                switch (k) {
                    case 0: return m_K_blk;
                    case 1: return m_K >= 2 * m_K_blk ? m_K_blk : 0;
                    case 2: return m_K_tail;
                    default:  IE_THROW() << "BrgemmEmitter detected unsupported K value";
                }
            }();
            if (brgemmCtx.N == 0 || brgemmCtx.N > m_N ||
                brgemmCtx.K == 0 || brgemmCtx.K > m_K)
                continue;

            brgemmCtx.LDA = leading_dimensions[0];
            brgemmCtx.LDB = brgemm_node->is_with_data_repacking() ? rnd_up(m_N, m_N_blk) : leading_dimensions[1];
            brgemmCtx.LDC = leading_dimensions[2];
            brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg0Prc));
            brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg1Prc));
            // Note: beta must be 1 ony if we need to add the result to the output
            brgemmCtx.beta = has_K_kernel ? 1 : 0;
//            std::cerr << "k = " << k << " n = " << n << " beta = " << brgemmCtx.beta << "\n";
            initBrgemm(brgemmCtx, m_brgKernels[kernel_idx], brgWithAMX);
            has_N_kernel = true;
        }
        if (has_N_kernel)
            has_K_kernel = true;
    }
    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> BrgemmEmitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "BrgemmEmitter::get_supported_precisions() expects BrgemmCPU node");
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
            throw ov::Exception("BrgemmEmitter got BrgemmCPU node with unsupported type");
    }
}

void BrgemmEmitter::initBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) {
    brgemm_t brgDesc;
//    brgemm_strides_t strides {static_cast<dnnl_dim_t>(ctx.M * ctx.K), static_cast<dnnl_dim_t>(ctx.K * ctx.N)};
    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_undef
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
//    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
//                                   false, false, brgemm_row_major, 1.f, 0.f, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, &strides);
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
                                   false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, nullptr);
    if (status != dnnl_success)
        IE_THROW() << "BrgemmEmitter cannot initialize brgemm descriptor due to invalid params";

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx)
        amx_tile_configure(ctx.palette);

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success)
        IE_THROW() << "BrgemmEmitter cannot create brgemm kernel due to invalid params";
    brgKernel.reset(brgKernel_);
}
size_t BrgemmEmitter::aux_gprs_count() const {
    return  m_N_blocking_loop_needed + m_K_blocking_loop_needed;
}
void BrgemmEmitter::emit_N_blocking_loops(size_t k_kernel_id,
                                          const Xbyak::Reg64& input_0, const Xbyak::Reg64& input_1,
                                          const Xbyak::Reg64& input_2, const Xbyak::Reg64& output_0,
                                          const Xbyak::Reg64& work_amount_N) const {
    auto emit_and_shift_pointers = [&](size_t kernel_idx) {
        const auto& brgemmCtx = m_brgCtxs[kernel_idx];
        emit_brgemm_kernel_call(m_brgKernels[kernel_idx].get(), brgemmCtx, input_0, input_1, input_2, output_0);
        h->add(output_0, brgemmCtx.N * io_data_size.back());
        h->add(input_1, brgemmCtx.N * io_data_size[1]);
        if (m_with_scratch && m_with_comp)
            h->add(input_2, brgemmCtx.N * io_data_size[2]);
    };
    // Blocked N loop
    size_t kernel_idx = getBrgIdx(k_kernel_id, 0);
    if (m_brgKernels[kernel_idx]) {
        const auto& brgemmCtx = m_brgCtxs[kernel_idx];
        Label N_loop_begin;
        if (m_N_blocking_loop_needed) {
            h->mov(work_amount_N, m_N);
            h->L(N_loop_begin);
        }

        emit_and_shift_pointers(kernel_idx);

        if (m_N_blocking_loop_needed) {
            h->sub(work_amount_N, brgemmCtx.N);
            h->cmp(work_amount_N, brgemmCtx.N);
            h->jge(N_loop_begin);
        }
    }
    // N loop tail
    kernel_idx = getBrgIdx(k_kernel_id, 1);
    if (m_brgKernels[kernel_idx])
        emit_and_shift_pointers(kernel_idx);
}
void BrgemmEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        Xbyak::Reg64 work_amount_N(static_cast<int>(0));
        Xbyak::Reg64 work_amount_K(static_cast<int>(0));
        if (m_N_blocking_loop_needed || m_K_blocking_loop_needed) {
            if (aux_gpr_idxs.size() < m_N_blocking_loop_needed + m_K_blocking_loop_needed)
                IE_THROW() << "BRGEMM Emitter requires extra gpr which was not allocated";
            if (m_N_blocking_loop_needed)
                work_amount_N = Xbyak::Reg64(static_cast<int>(aux_gpr_idxs[0]));
            if (m_K_blocking_loop_needed)
                work_amount_K = Xbyak::Reg64(static_cast<int>(aux_gpr_idxs[m_N_blocking_loop_needed]));
        }

        h->add(input_0, m_load_offset_a);
        h->add(input_1, m_load_offset_b);
        h->add(output_0, m_store_offset_c);
        if (m_with_scratch) {
            if (in.size() != 3)
                IE_THROW() << "BRGEMM Emitter expects 3 inputs if there are compensations/wsp";
            input_2 = Xbyak::Reg64(static_cast<int>(in[2]));
            h->add(input_2, m_load_offset_scratch);
        }
        // returns the first idx of non-empty K kernel, or -1 otherwise
        auto get_K_kernel_idx = [this](size_t k_kernel_id, size_t& kernel_idx) {
            for (size_t n = 0; n < BRGEMM_KERNELS_NUM[1]; n++) {
                const auto idx = getBrgIdx(k_kernel_id, n);
                if (m_brgKernels[idx]) {
                    kernel_idx = idx;
                    return true;
                }
            }
            return false;
        };
        // Blocked K loop
        const auto k_tail_id = BRGEMM_KERNELS_NUM[0] - 1;
        size_t total_K_work_amount = m_K;
        size_t kernel_idx = SIZE_MAX;
        for (size_t k_blocked_id = 0; k_blocked_id < k_tail_id; k_blocked_id++) {
            if (get_K_kernel_idx(k_blocked_id, kernel_idx)) {
                const auto& brgemmCtx = m_brgCtxs[kernel_idx];
                Label K_loop_begin;
                // Note: we never emit loop for the first blocked kernel, since it always executed only once.
                // The purpose of the first blocked K kernel is to initializes output, because it has beta = 0
                if (k_blocked_id == 0) {
                    total_K_work_amount -= brgemmCtx.K;
                } else if (m_K_blocking_loop_needed) {
                    h->mov(work_amount_K, total_K_work_amount);
                    h->L(K_loop_begin);
                }

                emit_N_blocking_loops(k_blocked_id, input_0, input_1, input_2, output_0, work_amount_N);
                h->sub(input_1, m_N * io_data_size[1]);
                h->sub(output_0, m_N * io_data_size.back());

                h->add(input_0, brgemmCtx.K * io_data_size[0]);
                h->add(input_1, (brgemmCtx.K * m_N) * io_data_size[1]);

                if (m_K_blocking_loop_needed && k_blocked_id) {
                    h->sub(work_amount_K, brgemmCtx.K);
                    h->cmp(work_amount_K, brgemmCtx.K);
                    h->jge(K_loop_begin);
                }
            }
        }
        // K loop tail
        if (get_K_kernel_idx(k_tail_id, kernel_idx)) {
            emit_N_blocking_loops(k_tail_id, input_0, input_1, input_2, output_0, work_amount_N);
            h->sub(input_1, m_N * io_data_size[1]);
            h->sub(output_0, m_N * io_data_size.back());
        }

        h->sub(input_0, m_load_offset_a + (m_K - m_K_tail) * io_data_size[0]);
        h->sub(input_1, m_load_offset_b + (m_K - m_K_tail) * m_N * io_data_size[1]);
        if (m_with_scratch)
            h->sub(input_2, m_load_offset_scratch);
        h->sub(output_0, m_store_offset_c);
    } else {
        IE_THROW() << "BrgemmEmitter requires at least avx512_core instruction set";
    }
}

void BrgemmEmitter::emit_brgemm_kernel_call(const brgemm_kernel_t *brg_kernel, const brgemmCtx& ctx,
                                            Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                            const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                            const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    if (ctx.is_with_amx) {
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // save function address in gpr to pass in call instruction
        const auto& overload = static_cast<status_t(*)(const char*)>(amx_tile_configure);
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(overload));
        h->mov(abi_param1, reinterpret_cast<uintptr_t>(ctx.palette));

        // align stack on 16-byte as ABI requires
        // note that RBX must not be changed by the callee
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        h->call(h->rbp);

        h->add(h->rsp, h->rbx);
        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }

    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
    for (size_t i = 0; i < n_k_regs_to_save; ++i) {
        if (mayiuse(avx512_core))
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
        else
            h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Zmm(i));

    size_t num_args_passed_on_stack = 0;
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
    num_args_passed_on_stack = 6;  // count of function brgemm_kernel_overload() parameters
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

    // align stack on 16-byte as ABI requires
    // note that RBX must not be changed by the callee
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    h->call(h->rbp);

    h->add(h->rsp, h->rbx);
    if (num_args_passed_on_stack > 0)
        h->add(h->rsp, num_args_passed_on_stack * gpr_size);
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        h->uni_vmovups(Zmm(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k registers
    for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
        if (mayiuse(avx512_core))
            h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        else
            h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
    }
    h->add(h->rsp, n_k_regs_to_save * k_mask_size);

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void BrgemmEmitter::kernel_execute(const brgemm_kernel_t *brg_kernel,
                                   const void *A, const void *B, void *C, void *scratch, int with_comp) {
    brgemm_kernel_params_t brgemm_p;
//    std::cerr << "A = " << A << " B = " << B << " C = " << C << "\n";
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
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

BrgemmCopyBEmitter::BrgemmCopyBEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(n);
    if (!brgemm_repack)
        IE_THROW() << "BrgemmCopyBEmitters expects BrgemmCopyB node";

    m_brgemm_prc_in0 = brgemm_repack->get_src_element_type();
    m_brgemm_prc_in1 = brgemm_repack->get_input_element_type(0);
    m_brgemmVNNIFactor = 4 / m_brgemm_prc_in0.size();
    m_with_comp = brgemm_repack->is_with_compensations();
    m_in_offset = brgemm_repack->get_offset_in();
    m_out_offset = brgemm_repack->get_offset_out();
    if (m_with_comp)
        m_comp_offset = brgemm_repack->get_offset_compensations();

    auto layout = ngraph::snippets::utils::get_node_output_layout(brgemm_repack->get_input_node_shared_ptr(0));
    const auto& original_shape = brgemm_repack->get_input_shape(0);
    auto transposed_shape = original_shape;
    size_t leading_dimension = *(original_shape.rbegin());
    if (!layout.empty()) {
        transposed_shape.resize(layout.size(), 1);
        for (size_t i = 0; i < layout.size(); ++i) {
            transposed_shape[i] = original_shape[layout[i]];
        }
        // The idea here is to find "2" (for 4D shapes) in the layout and multiply dimensions that are to the right
        // This implies that "3" is the last layout value, otherwise this layout is not supported.
        // counting from the end since shape could be prepended with ones
        const int64_t num_last_dims = layout.end() - std::find(layout.begin(), layout.end(), layout.size() - 2) - 1;
        if (layout.back() != layout.size() - 1 || num_last_dims < 1)
            IE_THROW() << "BrgemmRepackEmitter detected invalid layout values: " <<
                       "check that this shape + layout combination is schedulable";
        leading_dimension = std::accumulate(original_shape.end() - num_last_dims, original_shape.end(), 1, std::multiplies<size_t>());
    }

    m_N = *(transposed_shape.rbegin());
    m_K = *(transposed_shape.rbegin() + 1);

    const bool isAMXSupported = mayiuse(avx512_core_amx);
    const auto use_amx = isAMXSupported && m_brgemm_prc_in0 != ov::element::f32 && (m_K % m_brgemmVNNIFactor == 0) && (m_N % m_brgemmVNNIFactor == 0);

    m_N_blk = m_brgemm_prc_in1 == ov::element::f32 ? m_N :
              m_brgemm_prc_in1 == ov::element::bf16 ? 32 : 64;
    m_K_blk = use_amx ? m_brgemm_prc_in0 == ov::element::bf16 ? 32 : 64
                      : m_K;
    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;
    m_LDB = m_brgemm_prc_in1 == ov::element::f32 ? leading_dimension : rnd_up(m_N, m_N_blk);

    const auto dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(InferenceEngine::details::convertPrecision(m_brgemm_prc_in0)));
    const auto dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(InferenceEngine::details::convertPrecision(m_brgemm_prc_in1)));
    init_brgemm_copy(m_kernel, leading_dimension, m_N_blk, m_N_tail, m_LDB, m_K - m_K_tail, use_amx, dt_in0, dt_in1);
}

void BrgemmCopyBEmitter::init_brgemm_copy(std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                          size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                                          bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(N_blk);
    brgCopyKernelConf.wei_tag = dnnl_abcd;  // What's about other ranks?
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = static_cast<dim_t>(LDB);
    brgCopyKernelConf.N =  static_cast<dim_t>(N);
    brgCopyKernelConf.N_tail =  static_cast<dim_t>(N_tail);
    brgCopyKernelConf.N_blk =  static_cast<dim_t>(N_blk);
    brgCopyKernelConf.K =  static_cast<dim_t>(K);
    brgCopyKernelConf.K_blk =  static_cast<dim_t>(K);
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = avx512_core_amx;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    auto status = matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf);
    if (status != dnnl_success)
        IE_THROW() << "BrgemmRepackEmitter cannot create kernel due to invalid params";
}

void BrgemmCopyBEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 src(static_cast<int>(in[0]));
        Xbyak::Reg64 dst(static_cast<int>(out[0]));
        Xbyak::Reg64 comp(static_cast<int>(0));  // Compensations. Default reg idx is 0 if there aren't the compensations
        if (m_with_comp) {
            if (out.size() != 2) {
                IE_THROW() << "BrgemmCopyBEmitter with compensations requires separate register for them";
            }
            comp = Xbyak::Reg64(static_cast<int>(out[1]));
        }

        const size_t data_size = m_brgemm_prc_in1.size();
        for (size_t nb = 0; nb < div_up(m_N, m_N_blk); nb++) {
            const size_t offset_in = m_in_offset + nb * m_N_blk * data_size;
            const size_t offset_out = m_out_offset + nb * m_N_blk * m_brgemmVNNIFactor * data_size;
            const size_t offset_comp = m_with_comp ? m_comp_offset + nb * m_N_blk * sizeof(int32_t) : 0;

            const bool is_N_tail = (m_N - nb * m_N_blk < m_N_blk);
            const auto current_N_blk = is_N_tail ? m_N_tail : m_N_blk;

            emit_kernel_call(m_kernel.get(), src, dst, comp, current_N_blk, m_K, offset_in, offset_out, offset_comp);
        }
    } else {
        IE_THROW() << "BrgemmCopyBEmitter requires at least avx512_core instruction set";
    }
}

void BrgemmCopyBEmitter::emit_kernel_call(const matmul::jit_brgemm_matmul_copy_b_t* kernel, Reg64 src, Reg64 dst, Reg64 comp,
                                          size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) const {
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
    for (size_t i = 0; i < n_k_regs_to_save; ++i) {
        if (mayiuse(avx512_core))
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
        else
            h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Zmm(i));

    const auto data_ptr = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
#ifdef _WIN32
    const auto push_value = [&](size_t value, size_t index) {
        // Firstly we need to move integer to GPR. Then we can move value from GPR to stack
        h->mov(abi_not_param1, value);
        h->mov(h->qword[h->rsp + index * gpr_size], abi_not_param1);
    };
#endif

    size_t num_args_passed_on_stack = 0;
    // save function address in gpr to pass in call instruction
    const auto &kernel_overload = static_cast<void (*)(matmul::jit_brgemm_matmul_copy_b_t*,
                                                       const void*,
                                                       const void*,
                                                       const void*,
                                                       size_t,
                                                       size_t)>(execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), src);
    h->uni_vmovq(Xmm(1), dst);
    if (m_with_comp)
        h->uni_vmovq(Xmm(2), comp);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(kernel));

    data_ptr(Xmm(0), abi_param2, offset_in);
    data_ptr(Xmm(1), abi_param3, offset_out);
    if (m_with_comp) {
        data_ptr(Xmm(2), abi_param4, offset_comp);
    } else {
        h->mov(abi_param4, reinterpret_cast<uintptr_t>(nullptr));
    }

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    num_args_passed_on_stack = 6;  // count of function kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);

    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);
    push_value(N, abi_param_count + 0);
    push_value(K, abi_param_count + 1);
#else
    h->mov(abi_param5, N);
    h->mov(abi_param6, K);
#endif
    // align stack on 16-byte as ABI requires
    // note that RBX must not be changed by the callee
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    h->call(h->rbp);

    h->add(h->rsp, h->rbx);
    if (num_args_passed_on_stack > 0)
        h->add(h->rsp, gpr_size * num_args_passed_on_stack);
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        h->uni_vmovups(Zmm(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k registers
    for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
        if (mayiuse(avx512_core))
            h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        else
            h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
    }
    h->add(h->rsp, n_k_regs_to_save * k_mask_size);

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void BrgemmCopyBEmitter::execute(matmul::jit_brgemm_matmul_copy_b_t *kernel, const void *src,
                                 const void *dst, const void *comp, size_t N, size_t K) {
    if (!kernel)
        IE_THROW() << "Kernel for `brgemm_copy_b` hasn't been created";

    auto ctx = dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = N;
    ctx.src = src;
    ctx.tr_src = dst;
    ctx.compensation_ptr = comp;
    ctx.zp_a_compensation_ptr = nullptr;
    ctx.zp_a_neg_value_ptr = nullptr;
    ctx.current_K_start = 0;
    ctx.current_K_iters = K;

    (*kernel)(&ctx);
}

HorizonMaxEmitter::HorizonMaxEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n) :
    jit_emitter(h, isa, n, Precision::FP32, emitter_in_out_map::vec_to_vec) {}

void HorizonMaxEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "HorizonMax emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void HorizonMaxEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm src_vmm = Vmm(in[0]);
    Xmm dst_xmm = Xmm(out[0]);
    Xmm aux_xmm = Xmm(aux_vec_idxs[0]);

    Reg64 aux_reg = Reg64(aux_gpr_idxs[0]);

    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    const size_t vec_size = vlen / sizeof(float);
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], src_vmm);
    // Let the first value be the max
    h->mov(aux_reg, h->ptr[h->rsp]);
    h->vmovq(dst_xmm, aux_reg);
    for (size_t i = 1; i < vec_size; i++) {
        h->mov(aux_reg, h->ptr[h->rsp + i * sizeof(float)]);
        h->vmovq(aux_xmm, aux_reg);
        h->uni_vmaxps(dst_xmm, dst_xmm, aux_xmm);
    }
    h->add(h->rsp, vlen);
}

HorizonSumEmitter::HorizonSumEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n) :
    jit_emitter(h, isa, n, Precision::FP32, emitter_in_out_map::vec_to_vec) {}

void HorizonSumEmitter::emit_impl(const std::vector<size_t>& in,
                                  const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "HorizonSum emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void HorizonSumEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm src_vmm = Vmm(in[0]);
    Xmm dst_xmm = Xmm(out[0]);
    Xmm aux_xmm = Xmm(aux_vec_idxs[0]);

    Reg64 aux_reg = Reg64(aux_gpr_idxs[0]);

    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    const size_t vec_size = vlen / sizeof(float);
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], src_vmm);
    h->uni_vpxor(dst_xmm, dst_xmm, dst_xmm);
    for (size_t i = 0; i < vec_size; i++) {
        h->mov(aux_reg, h->ptr[h->rsp + i * sizeof(float)]);
        h->vmovq(aux_xmm, aux_reg);
        h->uni_vaddps(dst_xmm, dst_xmm, aux_xmm);
    }
    h->add(h->rsp, vlen);
}

VectorBufferEmitter::VectorBufferEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n) :
    jit_emitter(h, isa, n, Precision::FP32, emitter_in_out_map::vec_to_vec) {}

void VectorBufferEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Zero emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void VectorBufferEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm vmm = Vmm(out[0]);
    h->uni_vpxor(vmm, vmm, vmm);
}

FillEmitter::FillEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n) :
    jit_emitter(h, isa, n, Precision::FP32, emitter_in_out_map::vec_to_vec) {
    const auto fill = ov::as_type_ptr<ngraph::snippets::op::Fill>(n);
    if (fill->get_element_type().size() != 4) {
        IE_THROW() << "Fill emitter supports only 4 Byte element types but gets: " << fill->get_element_type();
    }

    offset = fill->get_offset();
    fill_value = fill->get_fill_value();
    prepare_table();
}

size_t FillEmitter::aux_gprs_count() const {
    // + 1 reg for temp reg for mask in avx512
    return one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core) ? 2 : 1;
}

void FillEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Fill emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void FillEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm src_vmm = Vmm(in[0]);
    Vmm dst_vmm = Vmm(out[0]);

    if (one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core)) {
        uint64_t tail_mask = 1;
        tail_mask = ~((tail_mask << offset) - tail_mask);
        h->mov(Reg64(aux_gpr_idxs[0]), tail_mask);
        h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
        h->vblendmps(dst_vmm | k_mask, src_vmm, table_val("value"));
    } else if (one_of(host_isa_, dnnl::impl::cpu::x64::avx2, dnnl::impl::cpu::x64::sse41)) {
        uint8 imm = 1;
        imm = ~((imm << offset) - imm);  // shift load_num bit
        if (host_isa_ == dnnl::impl::cpu::x64::sse41 && src_vmm.getIdx() != dst_vmm.getIdx()) {
            h->uni_vmovups(dst_vmm, src_vmm);
            src_vmm = Vmm(dst_vmm.getIdx());
        }
        h->uni_vblendps(dst_vmm, src_vmm, table_val("value"), imm);
    }
}

void FillEmitter::register_table_entries() {
    push_arg_entry_of("value", fill_value, true);
}

}   // namespace intel_cpu
}   // namespace ov
