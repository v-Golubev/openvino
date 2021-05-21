// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"

#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>
#include <low_precision/lpt_itt.hpp>

#include "low_precision/align_quantization_intervals.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/align_quantization_parameters.hpp"

// TODO: linker error in Windows
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"

// branch specific transformations
#include "low_precision/concat.hpp"

#include "low_precision/fake_quantize_decomposition.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/split.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::LowPrecision, "LowPrecision", 0);

ngraph::pass::low_precision::LowPrecision::LowPrecision(
    const std::vector<OperationPrecisionRestriction>& precisionRestrictions,
    const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions,
    const LayerTransformation::Params params) :
    precisionRestrictions(precisionRestrictions),
    quantizationRestrictions(quantizationRestrictions),
    params(params) {
}

using namespace ngraph::pass::low_precision;

template <typename BaseOp>
void make_matcher_type_relaxed(ngraph::pass::GraphRewrite* transformation) {
    using namespace ngraph;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!as_type_ptr<BaseOp>(n);
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto l_node = std::dynamic_pointer_cast<BaseOp>(m.get_match_root());
        if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(l_node)) {
            return false;
        }
        if (!l_node) {
            THROW_IE_LPT_EXCEPTION(*l_node) << "unexpected operation type";
        }

        std::vector<element::Type> inputPrecisions;
        for (auto& inputs : l_node->inputs()) {
            inputPrecisions.push_back(inputs.get_element_type());
        }

        std::vector<element::Type> outputPrecisions;
        for (auto& output : l_node->outputs()) {
            outputPrecisions.push_back(output.get_element_type());
        }

        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(*l_node, inputPrecisions, outputPrecisions);

        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "TypeRelaxedReplacer");
    NGRAPH_SUPPRESS_DEPRECATED_START
    transformation->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}

ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer::TypeRelaxedReplacer() {
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Clamp>(this);
    /// new Concat uses clone_with_new_inputs as result for TypeRelaxed we need to manage output precision manually
    /// just unwrap to TypeRelaxed
    // TODO: this update is absent in master
    //make_matcher_type_relaxed<opset1::Concat>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::ConvolutionBackpropData>(this);
    make_matcher_type_relaxed<opset1::DepthToSpace>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::PRelu>(this);
    make_matcher_type_relaxed<opset1::ReduceMean>(this);
    make_matcher_type_relaxed<opset1::ReduceSum>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<opset1::Interpolate>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
    make_matcher_type_relaxed<op::MVN>(this);
    make_matcher_type_relaxed<opset6::MVN>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
    make_matcher_type_relaxed<opset4::Interpolate>(this);
}

bool ngraph::pass::low_precision::LowPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto passConfig = get_pass_config();

    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionPrerequisites");
        ngraph::pass::Manager manager(passConfig);
        auto prerequisites = manager.register_pass<ngraph::pass::GraphRewrite>();
        const std::vector<ngraph::element::Type> supportedTypes = {ngraph::element::i8, ngraph::element::u8};
        prerequisites->add_matcher<PullReshapeThroughDequantization>(supportedTypes);
        prerequisites->add_matcher<PullTransposeThroughDequantization>(supportedTypes);
        prerequisites->add_matcher<ngraph::pass::LinOpSequenceFusion>();
        manager.run_passes(f);
    }

    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionStepTypeRelaxedReplacer");
        TypeRelaxedReplacer pass;
        pass.run_on_function(f);
    }

    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionStepCommon");

//#define VISUALIZE_TREE
#ifndef VISUALIZE_TREE
        {
            ngraph::pass::Manager markupAndDecompose(passConfig);
            markupAndDecompose.register_pass<low_precision::MarkupPrecisions>(precisionRestrictions);
            markupAndDecompose.register_pass<low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
            markupAndDecompose.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>();
            markupAndDecompose.register_pass<low_precision::PropagatePrecisions>();
            markupAndDecompose.register_pass<low_precision::AlignQuantizationIntervals>();
            markupAndDecompose.register_pass<low_precision::AlignQuantizationParameters>();
            markupAndDecompose.register_pass<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
            markupAndDecompose.run_passes(f);
        }
#else
//        #include <low_precision/align_quantization_parameters.hpp>
//        #include <low_precision/align_quantization_intervals.hpp>
//        #include <low_precision/markup_avg_pool_precision_preserved.hpp>
//        #include <low_precision/markup_per_tensor_quantization.hpp>
//        #include <low_precision/markup_precisions.hpp>
//        #include <low_precision/propagate_precisions.hpp>

        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.actual.svg").run_on_function(f);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.actual").run_on_function(f);

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::MarkupPrecisions>(precisionRestrictions);
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming1.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming1").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming2.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming2").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>();
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming3.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::PropagatePrecisions>();
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming4.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming4").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::AlignQuantizationIntervals>();
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming5.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming5").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::AlignQuantizationParameters>();
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming6.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming6").run_on_function(f);
        }

        {
            ngraph::pass::Manager tmp(passConfig);
            tmp.register_pass<low_precision::FakeQuantizeDecompositionTransformation>();
            tmp.run_passes(f);
            //ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming7.svg").run_on_function(f);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming7").run_on_function(f);
        }
#endif
        ngraph::pass::Manager manager(passConfig);
        std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
        common->add_matcher<ngraph::pass::low_precision::AddTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ClampTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::DepthToSpaceTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::InterpolateTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::GroupConvolutionTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::MatMulTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::MultiplyTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::MVNTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::NormalizeL2Transformation>(params);
        common->add_matcher<ngraph::pass::low_precision::PReluTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReduceMaxTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReduceMeanTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReduceMinTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReduceSumTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReluTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ReshapeTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::SqueezeTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::ShuffleChannelsTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::SplitTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::StridedSliceTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::TransposeTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::UnsqueezeTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::VariadicSplitTransformation>(params);
        manager.run_passes(f);
    }

    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionCleanup");
        {
            ngraph::pass::Manager cleanupManager(passConfig);
            std::shared_ptr<ngraph::pass::GraphRewrite> cleanup = cleanupManager.register_pass<ngraph::pass::GraphRewrite>();
            cleanup->add_matcher<ngraph::pass::low_precision::FoldConvertTransformation>(params);
            cleanup->add_matcher<ngraph::pass::low_precision::FuseConvertTransformation>(params);
            cleanupManager.run_passes(f);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager(passConfig);
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>(params);
            standaloneCleanupManager.run_passes(f);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager(passConfig);
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>(params);
            standaloneCleanupManager.run_passes(f);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager(passConfig);
            // WA: precision restrictions for groupConv must be propagated to MultiplyToGroupConvolution transformation
            const auto groupConvRestriction = OperationPrecisionRestriction::getPrecisionsByOperationType<opset1::GroupConvolution>(precisionRestrictions);
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>(params, groupConvRestriction);
            standaloneCleanupManager.run_passes(f);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager(passConfig);
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::SubtractMultiplyToMultiplyAddTransformation>(params);
            standaloneCleanupManager.run_passes(f);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager(passConfig);
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FoldFakeQuantizeTransformation>(params);
            standaloneCleanupManager.register_pass<ngraph::pass::ConstantFolding>();
            standaloneCleanupManager.run_passes(f);
        }
    }

    return true;
}

bool ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(const std::shared_ptr<ngraph::Function>& function) {
    std::set<std::shared_ptr<ngraph::Node>> handledNodes;
    std::deque<std::shared_ptr<ngraph::Node>> nodes;
    for (auto result : function->get_results()) {
        nodes.push_front(result);
    }

    while (!nodes.empty()) {
        auto node = nodes.front();
        nodes.pop_front();

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto parent = node->get_input_node_shared_ptr(i);
            if (handledNodes.find(parent) != handledNodes.end()) {
                continue;
            }

            const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
            if ((fakeQuantize != nullptr) &&
                QuantizationDetails::outputLayoutIsSupported(fakeQuantize) &&
                QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
                return true;
            }

            nodes.push_front(parent);
            handledNodes.insert(parent);
        }
    }
    return false;
}
