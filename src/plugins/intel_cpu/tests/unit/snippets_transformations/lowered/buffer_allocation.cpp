// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/validate_loops.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"

#include "transformations/snippets/x64/shape_inference.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/reduce_decomposition.hpp"
#include "transformations/snippets/x64/pass/lowered/set_brgemm_copy_b_buffers_shape.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"


namespace ov {
namespace test {
namespace snippets {

/*  Note[74841]:
 *  This test is almost full copy of BufferAllocationTest class from openvino/src/common/snippets/tests/include/lowered/pass/buffer_allocation.hpp.
 *  The BufferAllocationTest class should be shared test class to reuse this structure in backend-specific tests in test infrastructure refactoring.
 */

typedef std::tuple<ov::element::Type,  // MHA precision
                   bool,               // Optimized pipeline
                   bool,               // With SplitLoops opt
                   size_t,             // Expected Buffer size in bytes
                   size_t              // Expected unique Buffer IDs count
> BufferAllocationCPUParams;

class BufferAllocationCPUTest : public testing::TestWithParam<BufferAllocationCPUParams> {
public:
    using VectorDims = ov::snippets::VectorDims;
    static std::string getTestCaseName(testing::TestParamInfo<BufferAllocationCPUParams> obj) {
        ov::element::Type precision;
        bool is_optimized, with_split_loops;
        size_t expected_size, expected_count;
        std::tie(precision, is_optimized, with_split_loops, expected_size, expected_count) = obj.param;
        std::ostringstream result;
        result << "Opt=" << ov::test::utils::bool2str(is_optimized) << "_";
        result << "Split=" << ov::test::utils::bool2str(with_split_loops) << "_";
        result << "ExpBufferSize=" << expected_size << "_";
        result << "ExpBufferNum=" << expected_count;
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(m_precision, m_is_buffer_optimized, m_with_split_loops, m_expected_size, m_expected_count) = this->GetParam();

        const auto body = GetModel();
        m_linear_ir = ov::snippets::lowered::LinearIR(body, std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>());
        m_linear_ir.set_loop_depth(m_loop_depth);
        // When Subgraph::control_flow_transformations become public method,
        // please use this method instead of ApplyTransformations
        ApplyTransformations(GetPassConfig());
    }

    std::shared_ptr<ov::snippets::lowered::pass::PassConfig> GetPassConfig() {
        auto config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
        if (!m_with_split_loops)
            config->disable<ov::snippets::lowered::pass::SplitLoops>();
        return config;
    }

    void ApplyTransformations(const std::shared_ptr<ov::snippets::lowered::pass::PassConfig>& pass_config) {
        ov::snippets::lowered::pass::PassPipeline pipeline(pass_config);
        pipeline.register_pass<ov::snippets::lowered::pass::MarkLoops>(m_vector_size);
        pipeline.register_pass<ov::intel_cpu::pass::BrgemmBlocking>();
        pipeline.register_pass<ov::intel_cpu::pass::ReduceDecomposition>(m_vector_size);
        pipeline.register_pass<ov::snippets::lowered::pass::FuseLoops>();
        pipeline.register_pass<ov::snippets::lowered::pass::SplitLoops>();
        pipeline.register_pass<ov::snippets::lowered::pass::InsertBuffers>(2);
        pipeline.register_pass<ov::snippets::lowered::pass::InsertLoadStore>(m_vector_size);
        pipeline.register_pass<ov::snippets::lowered::pass::InitLoops>();
        pipeline.register_pass<ov::snippets::lowered::pass::InsertLoops>();
        pipeline.register_pass<ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape>();
        pipeline.register_pass<ov::snippets::lowered::pass::AllocateBuffers>(m_buffer_scratchpad, m_is_buffer_optimized);
        pipeline.run(m_linear_ir);
    }

    void Validate() {
        std::set<size_t> gprs;
        for (const auto& expr : m_linear_ir) {
            if (const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(expr->get_node())) {
                gprs.insert(buffer->get_id());
            }
        }
        EXPECT_EQ(gprs.size(), m_expected_count);
        EXPECT_EQ(m_buffer_scratchpad, m_expected_size);
    }

    virtual std::shared_ptr<ov::Model> GetModel() const = 0;

    void MarkOp(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& subtensor) const {
        for (const auto& input : node->inputs())
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
                input, std::make_shared<ov::snippets::lowered::PortDescriptor>(input, subtensor));
        for (const auto& output : node->outputs())
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
                output, std::make_shared<ov::snippets::lowered::PortDescriptor>(output, subtensor));
    }

    ov::element::Type m_precision;

    size_t m_buffer_scratchpad = 0;
    ov::snippets::lowered::LinearIR m_linear_ir;

    size_t m_expected_size = 0;
    size_t m_expected_count = 0;

    size_t m_loop_depth = 2;
    size_t m_vector_size = 16;

    bool m_is_buffer_optimized = true;
    bool m_with_split_loops = true;
};

class MHABaseBufferAllocationTest : public BufferAllocationCPUTest {
protected:
    virtual std::shared_ptr<ov::intel_cpu::BrgemmCPU> GetBrgemm(const ov::Output<ov::Node>& input0,
                                                                const ov::Output<ov::Node>& input1,
                                                                const std::vector<size_t>& subtensor) const = 0;
    std::shared_ptr<ov::Model> GetModel() const override {
        const auto subtensor_scalar = std::vector<size_t>{1};
        const auto subtensor_softmax = std::vector<size_t>{1, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM);

        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(m_precision, ov::PartialShape({1, 12, 128, 64}));
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(m_precision, ov::PartialShape({1, 128, 12, 64}));
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(m_precision, ov::PartialShape({1, 12, 128, 64}));

        const auto load_reshape = std::make_shared<ov::snippets::op::LoadReshape>(parameter1, 1, 0, std::vector<size_t>{0, 2, 3, 1});
        const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
        const auto convert0 = std::make_shared<ov::snippets::op::ConvertSaturation>(store, ov::element::f32);
        const auto relu0 = std::make_shared<ov::op::v0::Relu>(convert0);
        const auto convert1 = std::make_shared<ov::snippets::op::ConvertSaturation>(relu0, m_precision);
        const auto brgemm_cpu0 = GetBrgemm(parameter0, convert1, subtensor_full);

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu0);

        // Decomposed Softmax
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 3);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 3);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        const auto convert2 = std::make_shared<ov::snippets::op::ConvertSaturation>(multiply, m_precision);
        const auto brgemm_cpu1 = GetBrgemm(convert2, parameter2, subtensor_full);

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu2), ov::ParameterVector{parameter0, parameter1, parameter2});

        MarkOp(load_reshape, subtensor_scalar);
        MarkOp(store, subtensor_scalar);
        MarkOp(reduce_max, subtensor_softmax);
        MarkOp(reduce_sum, subtensor_softmax);
        MarkOp(power, subtensor_softmax);

        return body;
    }
};

class MHABufferAllocationTest : public MHABaseBufferAllocationTest {
protected:
    std::shared_ptr<ov::intel_cpu::BrgemmCPU> GetBrgemm(const ov::Output<ov::Node>& input0,
                                                        const ov::Output<ov::Node>& input1,
                                                        const std::vector<size_t>& subtensor) const override {
        const auto brgemm_cpu = std::make_shared<ov::intel_cpu::BrgemmCPU>(input0, input1, ov::intel_cpu::BrgemmCPU::Type::Floating);
        const auto& b_shape = input1.get_partial_shape();
        const auto& k_dimension = b_shape[b_shape.size() - 2];
        const auto& n_dimension = b_shape[b_shape.size() - 1];
        OPENVINO_ASSERT(k_dimension.is_static() && n_dimension.is_static(), "MHABufferAllocationTest supports brgemm creation only with static k & n dims.");

        brgemm_cpu->set_m_block_size(32);
        brgemm_cpu->set_k_block_size(k_dimension.get_length());
        brgemm_cpu->set_n_block_size(n_dimension.get_length());
        MarkOp(brgemm_cpu, subtensor);
        return brgemm_cpu;
    }
};

class MHABF16AMXBufferAllocationTest : public MHABaseBufferAllocationTest {
protected:
    std::shared_ptr<ov::intel_cpu::BrgemmCPU> GetBrgemm(const ov::Output<ov::Node>& input0,
                                                        const ov::Output<ov::Node>& input1,
                                                        const std::vector<size_t>& subtensor) const override {
        const auto brgemm_copyb = std::make_shared<ov::intel_cpu::BrgemmCopyB>(
            input1, ov::element::bf16, ov::intel_cpu::BrgemmCopyB::OnlyRepacking, 0, 0, 0);
        const auto scratch =
            std::make_shared<ov::snippets::op::NewMemoryBuffer>(ov::Shape{ov::intel_cpu::BrgemmCPU::SCRATCH_BYTE_SIZE});
        const auto brgemm_cpu = std::make_shared<ov::intel_cpu::BrgemmCPU>(
            input0, brgemm_copyb->output(0), scratch, ov::intel_cpu::BrgemmCPU::Type::AMX);
        brgemm_cpu->set_m_block_size(32);
        brgemm_cpu->set_k_block_size(16);
        brgemm_cpu->set_n_block_size(64);
        MarkOp(brgemm_cpu, subtensor);
        MarkOp(brgemm_copyb, subtensor);
        MarkOp(scratch, subtensor);
        return brgemm_cpu;
    }
};

TEST_P(MHABufferAllocationTest, BufferAllocationCPU) {
    Validate();
}

TEST_P(MHABF16AMXBufferAllocationTest, BufferAllocationCPU) {
    Validate();
}


namespace BufferAllocationCPUTest_Instances {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(139264), // Each Buffer has own allocated memory
                                 ::testing::Values(7)),  // Each Buffer has unique ID
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(57344), // (Buffer before brgemm) + (between brgemms) + (after brgemm)
                                 ::testing::Values(2)), // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(360448), // Each Buffer has own allocated memory
                                 ::testing::Values(7)),  // Each Buffer has unique ID
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(98304), // (between brgemms) + (Buffer before brgemm0 and after brgemm1)
                                 ::testing::Values(2)), // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXNotOptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(196608),
                                 ::testing::Values(11)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16OptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(90112),
                                 ::testing::Values(3)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16NotOptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(393216),
                                 ::testing::Values(11)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16OptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(114688),
                                 ::testing::Values(4)),
                         BufferAllocationCPUTest::getTestCaseName);

}  // namespace BufferAllocationCPUTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov
