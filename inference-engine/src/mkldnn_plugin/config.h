// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_istreams_executor.hpp>
#include <ie_performance_hints.hpp>
#include "utils/debug_capabilities.h"

#include <string>
#include <map>

namespace MKLDNNPlugin {

struct Config {
    Config();

    enum LPTransformsMode {
        Off,
        On,
    };

    enum TokenizationMode {
        Disabled,
        Node,
        Subgraph
    };

    enum FusionsMode {
        TargetIndependant = 0x1,
        TargetDependant = 0x2,
        All = TargetIndependant | TargetDependant,
        None = 0x0
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    int batchLimit = 0;
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
#if defined(__arm__) || defined(__aarch64__)
    // Currently INT8 mode is not optimized on ARM, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
    bool enforceBF16 = false;
#else
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
    bool enforceBF16 = true;
    bool manualEnforceBF16 = false;
#endif

#ifdef CPU_DEBUG_CAPS
    DebugCaps::Config debugCaps;
#endif
    TokenizationMode tokenizationMode = TokenizationMode::Subgraph;
    FusionsMode fusionsMode = FusionsMode::All;

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;
};

}  // namespace MKLDNNPlugin
