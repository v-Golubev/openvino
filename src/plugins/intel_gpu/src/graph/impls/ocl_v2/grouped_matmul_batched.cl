/*******************************************************************************
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "include/batch_headers/common.cl"
#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"
#define DECORATOR gm
#include "expert_gemm_common.cl"
#include "expert_gemm_compute.cl"

// Grouped MatMul batched GEMM: no sort/gather, no scattered write-back.
// Tokens are already partitioned by expert in A, so we write output sequentially.
// Dispatch: x=M tiles, y=token tiles, z=G groups (one expert per group).

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(grouped_matmul_batched)(
    OPTIONAL_SHAPE_INFO_ARG
    const global INPUT0_TYPE* input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
    const global uchar* weight_ptr,
#else
    const global INPUT1_TYPE* weight_ptr,
#endif
    global OUTPUT_TYPE* out_ptr,
    // Group metadata: for GroupedMatMul the group index == expert index.
    // group_expert_ids[g] = g   (identity)
    // group_offsets[g]    = cumulative start row of group g in A
    // group_sizes[g]      = number of tokens in group g
    const global int* group_expert_ids,
    const global int* group_offsets,
    const global int* group_sizes,
    const global int* num_groups,
    int m,
    int k
#ifdef BIAS_DT
    ,
    const global BIAS_DT* bias_ptr
#endif
#ifdef WEIGHT_COMPRESSED_INT4
    ,
    const global WEIGHT_SCALE_DT* weight_scales
#    ifdef WEIGHT_ZP_DT
    ,
    const global WEIGHT_ZP_DT* weight_zps
#    endif
#endif
#ifdef USE_SLM
    ,
    local int* slm
#endif
) {
    uint group_id = get_group_id(2);
    if (group_id >= (uint)num_groups[0])
        return;

    int expert_id = group_expert_ids[group_id];
    int offset    = group_offsets[group_id];    // start row in A (and in output)
    int cur_n_tokens = group_sizes[group_id];

    const global INPUT0_TYPE* group_input_ptr = input_ptr + offset * k;

    UGEMM_C_TYPE_HALF c_tile_half;
    uint sg_i0, sg_j0;
    if (!expert_gemm_compute(group_input_ptr, weight_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
                             weight_scales,
#    ifdef WEIGHT_ZP_DT
                             weight_zps,
#    endif
#endif
#ifdef BIAS_DT
                             bias_ptr,
#endif
#ifdef USE_SLM
                             slm,
#endif
                             expert_id, cur_n_tokens, m, k,
                             &c_tile_half, &sg_i0, &sg_j0))
        return;

    // Sequential store: output rows are already in-order — no token_map lookup.
    {
        int sglid = get_sub_group_local_id();
        const int br = UGEMM_C_TYPE_BLOCK0;
        const int nbr = UGEMM_C_TYPE_NBLOCK0;
        const int bc = UGEMM_C_TYPE_BLOCK1;
        const int nbc = UGEMM_C_TYPE_NBLOCK1;
        int sg = SUBGROUP_SIZE;

        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                // out[offset + sg_j0 + j, :] — token is already at its final row
                global OUTPUT_TYPE* row_ptr = out_ptr + (offset + sg_j0 + j) * m;
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        row_ptr[sg_i0 + i] = c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br) / sg + (j % bc) * (br / sg)];
                    }
                }
            }
        }
    }
}
