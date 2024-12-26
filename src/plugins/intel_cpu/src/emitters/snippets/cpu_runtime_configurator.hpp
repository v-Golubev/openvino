// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/jit_snippets_call_args.hpp"

#ifndef OPENVINO_ARCH_ARM64
#    include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#endif

#include "cache/multi_cache.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    OPENVINO_RTTI("CPURuntimeConfig", "0", ov::snippets::RuntimeConfig)
    CPURuntimeConfig() = default;

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

#ifndef OPENVINO_ARCH_ARM64
    struct RepackedInput {
        RepackedInput() = default;
        RepackedInput(std::shared_ptr<const BrgemmCopyBKernel> kernel,
                      CpuBlockedMemoryDescPtr desc,
                      VectorDims in_offsets,
                      VectorDims out_offsets);

        const std::shared_ptr<const BrgemmCopyBKernel>& kernel() const;
        const CpuBlockedMemoryDescPtr& desc() const;
        const VectorDims& in_offsets() const;
        const VectorDims& out_offsets() const;

    private:
        std::shared_ptr<const BrgemmCopyBKernel> m_kernel{nullptr};
        CpuBlockedMemoryDescPtr m_desc{nullptr};
        VectorDims m_in_offsets{};
        VectorDims m_out_offsets{};
    };
    std::unordered_map<size_t, RepackedInput> repacked_inputs = {};

    enum class RepackingImplType {
        NONE,         // no kernel-outside repacking
        IN_PARALLEL,  // should be executed in parallel_nt by each thread
        SEPARATE,     // should be separathy from kernel executed
    };
    RepackingImplType repacking_impl_type = RepackingImplType::NONE;
#endif  // OPENVINO_ARCH_ARM64

    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator(ov::intel_cpu::MultiCacheWeakPtr cache = {});

    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const;

    const ov::intel_cpu::MultiCacheWeakPtr& get_cache() const {
        return compiled_kernel_cache;
    }

protected:
    void update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;
    void update_tensor_rank(const ov::snippets::VectorDims& master_shape) const override;
    void init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const override;
    void initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;

    static const size_t rank6D;

    ov::intel_cpu::MultiCacheWeakPtr compiled_kernel_cache;
};

}  // namespace intel_cpu
}  // namespace ov
