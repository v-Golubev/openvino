# Plan: Non-Compressed f16 MoE for GPU (3-GEMM only)

## TL;DR
Add f16 non-compressed MoE weight support for the GPU plugin's fused 3-GEMM kernel. The kernel already has an f16 code path (`WEIGHT_COMPRESSION_DT=2`) but it had OCL compilation bugs, the OneDNN dispatch didn't handle f16 weights, and there was no transformation to route non-compressed MOE through the fused pipeline. Work bottom-up: fix and test the kernel primitive first, then the transformation, then functional tests.

## Background
1. `ConvertTiledMoeBlockTo3GatherMatmuls` creates `GatherMatmul` (4 inputs) or `GatherMatmulCompressed` (6 inputs)
2. `Convert3GatherMatmulMoeBlockToMoeOp` fuses routing + 3 GatherMatmul into plain `MOE` (6 inputs) or `MOECompressed` (12 inputs)
3. `FuseMOE3GemmCompressed` (GPU-only) fuses router MatMul + `MOECompressed` into `MOE3GemmFusedCompressed`
4. `CreateMOE3GemmFusedCompressedOp` maps to `moe_3gemm_fused_compressed` GPU primitive
5. OCL kernel dispatches with `WEIGHT_COMPRESSION_DT` = 0 (u4), 1 (u8), or 2 (f16)

**Gap**: Step 2 creates plain `MOE` for non-compressed weights. Step 3 only matches `MOECompressed`. Step 5 had bugs in the f16 path.

## Architecture: Execution Paths

The `moe_3gemm_fused_compressed` primitive has three execution paths:
- **Single-token path** (`token_num == 1`): Uses OCL kernels (`moe_3gemm_swiglu_mlp.cl`) that dispatch via `WEIGHT_COMPRESSION_DT`
- **Per-expert OneDNN loop** (`exec_prefill_onednn`): Creates `onednn_linear` per expert via `init_dnnl_weights()` + `get_kernel()`
- **Grouped OneDNN GEMM** (`exec_prefill_grouped_gemm`, default for `token_num > 1`): Uses grouped matmul primitives via `get_grouped_kernel()`

All three paths must handle f16 weights (no scale/zp).

## Strict Order of Work

**Rule**: Each layer must pass its unit tests before moving to the next layer.

### Phase 1: Kernel Primitive f16 Support — ✅ DONE
### Phase 2: Transformation — ✅ DONE
### Phase 3: GPU Functional Tests — ❌ IN PROGRESS (see below)

---

## Phase 1: Kernel Primitive (`moe_3gemm_fused_compressed`) — ✅ ALL PASSED

All 6 f16 kernel unit tests pass. All 50 existing compressed kernel tests pass.

### Changes Applied

**`moe_3gemm_swiglu_mlp.cl`** — OCL kernel fixes:
1. Fixed `gate_up_gemv_n2x_f16` call sites: removed extra `up_zp`/`gate_zp` args (was 7 args, function takes 6)
2. Set `expert_scale_size = 0` and `expert_zp_size = 0` for `WEIGHT_COMPRESSEION_DT == 2` in both gate_up and down kernels (prevents OOB pointer arithmetic on dummy scale/zp buffers)
3. Fixed address space qualifiers: `(const __global ushort*)x2` → `(const __local ushort*)x2` in `gate_up_gemv_n2x_f16` (both SUBGROUP_SIZE==32 and ==16 branches) and `down_gemv_n2x_f16` (SUBGROUP_SIZE==32 branch only; ==16 branch was already correct). This was the ROOT CAUSE of `CL_OUT_OF_RESOURCES` — `x2` is `__local half*` but was being cast to `__global` for sub-group block reads.

**`moe_3gemm_swiglu_opt.cpp`** — OneDNN f16 handling:
- `init_dnnl_weights()`: Detects `is_f16_weight`, sets `ic_group_size = -1`, skips scale/zp `convert2dnnl()` and cross-checks
- `get_grouped_kernel()`: Skips `attr.set_scales()`/`attr.set_zero_points()` for f16, skips scale/zp md creation
- `exec_prefill_grouped_gemm()`: Skips `DNNL_ARG_ATTR_SCALES` in GEMM args for f16

**`moe_3gemm_swiglu_opt.hpp`** — Kernel validator:
- Added `ov::element::f16` to `supported_wei_type`
- Early-return `true` for f16 (skips scale/zp validation)

**`moe_3gemm_gpu_test.cpp`** — f16 kernel unit tests:
- `moe_3gemm_f16_gpu_random` test class with `Moe3GemmF16TestParams`
- 6 test cases: SOFTMAX×3 + SIGMOID_BIAS×3
- All 6 PASS ✅

### Verification
```bash
cd build/Release && cmake --build . --target ov_gpu_unit_tests -j$(nproc)
cd ../../ && ./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="smoke/moe_3gemm_f16_gpu_random.*"
# [PASSED] 6 tests.
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*moe_3gemm_compressed_gpu*"
# [PASSED] 50 tests.
```

---

## Phase 2: Transformation (`FuseMOE3GemmCompressed`) — ✅ ALL PASSED

All 11 transformation tests pass (7 existing + 4 new).

### Changes Applied

**`fuse_moe_3gemm_compressed.hpp`** — Added `bool m_has_batch_dim = false;` member and constructor param

**`fuse_moe_3gemm_compressed.cpp`** — Extended to match plain MOE ops:
- Added pattern for `Constant(f16)->Convert(f32)` weight pattern
- Creates dummy scale/zp constants: `Constant::create(element::f16, Shape{0}, {})` ← **CRITICAL: must be `element::f16`, NOT `element::dynamic`**
  - `element::dynamic` causes jitter.cpp:324 to throw "unsupported data type" when compiling JIT constants for the GPU kernel, which is silently caught in `add_stage` (primitive_ocl_base.hpp:63-67), resulting in "Kernel not found in cache" errors
- Builds MOE3GemmFusedCompressed with config: `group_size = SIZE_MAX`, `has_zp = false`, `out_type = f16`

**`transformations_pipeline.cpp`** — `FuseMOE3GemmCompressed(has_batch_dim)` registration

**`fuse_moe_3gemm_compressed_test.cpp`** — 4 new `FuseMOE3GemmNonCompressedTest` test cases:
- Uses `element::f16` for dummy constants (same as transformation)
- All 4 PASS ✅

### Verification
```bash
cd build/Release && cmake --build . --target ov_gpu_unit_tests -j$(nproc)
cd ../../ && ./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*FuseMOE3Gemm*"
# [PASSED] 11 tests.
```

---

## Phase 3: GPU Functional Tests — ❌ REMAINING WORK

### Changes Already Applied

**`moe.cpp`** (functional tests):
- `MoENonCompressedTest` class added
- `smoke_MoE3GemmNonCompressed` (SWISH) and `smoke_MoE3GemmNonCompressedGelu` (GELU) instantiated
- Thresholds: `abs_threshold = 10.0`, `rel_threshold = 0.02`

### Current Test Results (8 tests, all FAIL)

**4 SWISH tests — Comparison failures (large numerical diffs):**
```
256/Softmax/SWISH:     Coord 3613  Expected: -225.625  Actual: -252.5     Diff: 26.875
256/SigmoidBias/SWISH: Coord 1281  Expected: 43008     Actual: inf        Diff: inf
128/Softmax/SWISH:     Coord 1308  Expected: 73.8125   Actual: 54.2812    Diff: 19.53
128/SigmoidBias/SWISH: Coord 768   Expected: -30.8281  Actual: 0          Diff: 30.83
```
These are NOT threshold issues — the diffs include `inf` and zero-vs-nonzero, indicating a **correctness bug** in the end-to-end pipeline. The kernel unit tests pass (reference matches), so the issue is likely in how the functional test model differs from the unit test setup — probably in the transformation output or weight layout assumptions.

**Key observations:**
- `Actual: inf` → f16 overflow, likely a weight or intermediate value getting too large
- `Actual: 0` vs `Expected: -30.83` → some values are not being written at all
- The kernel unit tests use small shapes (128/256 hidden, 256/512 inter) and pass, so the kernel itself is correct
- The functional test goes through `initMoE3GeMMSubgraph()` → full transformation pipeline → compilation → execution
- Something in the model structure or weight handling during the full pipeline is wrong

**Root cause hypotheses to investigate:**
1. Weight data range: `initMoE3GeMMSubgraph` uses `make_constant(f16, ..., InputGenerateData(0, 10, 1, seed))` — values 0-10 in f16. After matmul accumulation (K=128/256), values can reach ~1280-2560. After SwiGLU (multiply two matmul outputs), values can reach ~3.3M-6.5M which overflows f16 (max ~65504). The `inf` result confirms this.
2. The reference computation uses f32, which doesn't overflow, but the GPU kernel uses f16 throughout → overflow to inf
3. For the `Actual: 0` case, it could be that some expert's output underflows after routing weight multiplication, or some routing path is broken

**Possible fixes:**
- Reduce input data range (e.g., `InputGenerateData(0, 1, 1, seed)` or use normal distribution with small stddev) in the test's `generate_inputs` or adjust the model builder
- Or increase thresholds dramatically and accept f16 precision losses
- Or investigate if there's a real bug in how the transformation connects weight constants

**4 GELU tests — "GatherMatmul(extension) is not supported":**
The GELU activation is not supported by the fused 3-GEMM kernel. With compressed weights, GELU falls back to `GatherMatmulCompressed` which IS supported. But with non-compressed f16 weights, the fallback creates `GatherMatmul` (not compressed), which the GPU plugin does NOT support. **Action: Remove GELU test instantiation entirely** (or GTEST_SKIP when act==GELU).

### Next Steps for Phase 3

1. **Remove GELU tests** — `GatherMatmul` doesn't support non-compressed f16
2. **Investigate SWISH comparison failures** — Need to determine if this is:
   - (a) f16 overflow due to weight data range (test-only issue → fix test data generation)
   - (b) Real correctness bug in transformation/pipeline (needs code fix)
3. **Verify existing compressed functional tests still pass** after all changes

### Build & Test Commands
```bash
# Build
cd build/Release && cmake --build . --target ov_gpu_unit_tests -j$(nproc)
cd build/Release && cmake --build . --target ov_gpu_func_tests -j$(nproc)

# Phase 1 gate (kernel unit tests):
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="smoke/moe_3gemm_f16_gpu_random.*"
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*moe_3gemm_compressed_gpu*"

# Phase 2 gate (transformation unit tests):
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*FuseMOE3Gemm*"

# Phase 3 gate (functional tests):
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter="*smoke_MoE3GemmNonCompressed*"
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter="*smoke_MoE3GemmCompressedFusion*"

# Kill zombies:
pkill -9 -f ov_gpu_unit_tests 2>/dev/null; pkill -9 -f ov_gpu_func_tests 2>/dev/null
```

---

## All Modified Files (Current State)

### ✅ Kernel/Primitive (Phase 1 — complete, all tests pass)
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl` — OCL kernel f16 fixes (call args, scale/zp sizes, address space qualifiers)
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` — OneDNN f16 handling (init_dnnl_weights, get_grouped_kernel, exec_prefill_grouped_gemm)
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp` — f16 in supported_wei_type + early return
- `src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp` — f16 kernel unit tests (6 tests)

### ✅ Transformation (Phase 2 — complete, all tests pass)
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.hpp` — `has_batch_dim` parameter
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp` — Plain MOE matching pattern (uses `element::f16` for dummy constants)
- `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp` — `FuseMOE3GemmCompressed(has_batch_dim)` registration
- `src/plugins/intel_gpu/tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp` — 4 NonCompressed tests (uses `element::f16` for dummy constants)

### ❌ Functional Tests (Phase 3 — needs fixes)
- `src/plugins/intel_gpu/tests/functional/subgraph_tests/dynamic/moe.cpp` — MoENonCompressedTest class + instantiations (SWISH: comparison failures; GELU: unsupported op)

---

## Critical Lessons Learned

1. **`element::dynamic` is NOT valid for GPU layouts** — causes jitter.cpp:324 to throw, silently caught in `add_stage`. Use `element::f16` for dummy scale/zp constants.
2. **OCL address space qualifiers matter** — `x2` is `__local half*` but f16 kernels had `(const __global ushort*)x2` casts for sub-group block reads. Must use `__local`. The u4/u8 paths correctly used `__local`. This caused `CL_OUT_OF_RESOURCES` at runtime.
3. **Zombie GPU processes** — A hung `ov_gpu_unit_tests` consuming 100% CPU blocks all subsequent GPU test runs. Always `pkill -9` before testing.
4. **`add_stage` silently swallows exceptions** — `primitive_ocl_base.hpp:63-67` catches exceptions from `get_kernel_data()` and only logs at DEBUG level. If all stages fail, the kernel is never compiled, and you get "kernel not found in cache" at runtime. To debug: temporarily add `std::cerr` in the catch block.
5. **`onednn_matmul` with `ic_group_size = -1`** — Setting `ic_group_size = -1` tells the constructor to skip `w_scale()` and `w_zp()` attribute setup entirely. This is the correct way to handle non-compressed f16 weights in OneDNN paths.

## Decisions
- Scope: 3-GEMM only; 2-GEMM excluded
- **Bottom-up order**: kernel primitive → transformation → functional tests
- **Test-first**: write tests, verify they fail for the right reason, then fix
- **Verify after each fix**: run tests immediately, don't batch
- Extend GPU-specific `FuseMOE3GemmCompressed` (not common `moe_op_fusion.cpp`)
- Match `Constant(f16)->Convert(f32)` explicitly for plain MOE weights
- All dummy scale/zp inputs: `Constant(element::f16, Shape{0})` in transformation; `{1,1,1,1}` f16 memory in kernel unit tests
- Reuse existing `moe_3gemm_fused_compressed` kernel with `WEIGHT_COMPRESSION_DT=2`
- `has_batch_dim` passed as constructor parameter to `FuseMOE3GemmCompressed`
