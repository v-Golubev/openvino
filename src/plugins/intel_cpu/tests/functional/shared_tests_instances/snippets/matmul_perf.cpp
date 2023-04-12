// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul_perf.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<std::vector<ov::PartialShape>> input_shapes{
//          {{10, 18, 512, 64}, {10, 1, 64, 9216}},
//          {{10, 18, 512, 9216}, {10, 1, 9216, 64}},
          {{1, 40, 576, 64}, {1, 40, 64, 77}},
          {{1, 40, 576, 77}, {1, 40, 77, 64}},
//        {{2, 1, 3, 5}, {1, 3, 5, 3}},
//        {{3, 1, 32, 14}, {1, 2, 14, 32}},
//        {{1, 2, 37, 23}, {2, 1, 23, 37}},
//        {{1, 1, 37, 23}, {1, 2, 23, 33}},
//        {{1, 16, 384, 64}, {1, 16, 64, 384}}
};
static inline std::vector<std::vector<element::Type>> precisions(bool only_fp32 = true) {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32},
    };
    if (!only_fp32) {
        // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
        if (InferenceEngine::with_cpu_x86_avx512_core_vnni() || InferenceEngine::with_cpu_x86_avx512_core_amx_int8()) {
            prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
            prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
        }
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
    return prc;
}
//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMultPerf, MatMulPerf,
//                         ::testing::Combine(
//                             ::testing::ValuesIn(input_shapes),
//                             ::testing::ValuesIn(precisions(true)),
//                             ::testing::Values(1), // MatMul
//                             ::testing::Values(1), // Tokenized MatMul
//                             ::testing::ValuesIn({true, false}),
//                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                         MatMulPerf::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose = {
        {{1, 10, 63, 32}, {1, 10, 32, 32}, {1, 10, 32, 32}}
//        {{1, 40, 576, 64}, {1, 40, 64, 77}, {1, 40, 77, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposePerf, MHAWOTransposePerf,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::ValuesIn({ false}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::ValuesIn({true}),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHAWOTransposePerf::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov