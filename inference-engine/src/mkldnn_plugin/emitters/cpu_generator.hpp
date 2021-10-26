// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/generator.hpp"

namespace MKLDNNPlugin {

class CPUTarget : public ngraph::snippets::TargetMachine {
public:
    CPUTarget(dnnl::impl::cpu::x64::cpu_isa_t host_isa);

    bool is_supported() const override;
    ngraph::snippets::code get_snippet() const override;
    size_t get_lanes() const override;

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
};

class CPUGenerator : public ngraph::snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa);
    ~CPUGenerator() = default;
};

} // namespace MKLDNNPlugin