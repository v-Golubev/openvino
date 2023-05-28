// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/utils.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>


namespace {
auto is_supported_tensor(const ngraph::descriptor::Tensor& t) -> bool {
    return ngraph::snippets::utils::one_of(t.get_element_type(), ov::element::f32, ov::element::bf16) &&
           t.get_partial_shape().is_static() && t.get_shape().size() == 4;
}

auto is_supported_intermediate_op(const std::shared_ptr<ngraph::Node>& node) -> bool {
    const auto is_intermediate_op = [](const std::shared_ptr<ngraph::Node>& node) {
        return ngraph::is_type<ngraph::op::util::UnaryElementwiseArithmetic>(node) ||
               ngraph::is_type<ngraph::op::util::BinaryElementwiseArithmetic>(node) ||
               ngraph::is_type<ngraph::op::v1::Select>(node);
    };
    return ngraph::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(node) && is_intermediate_op(node);
}

auto is_valid_transpose(const std::shared_ptr<ngraph::opset1::Transpose>& node, std::vector<int64_t> expected_order) -> bool {
    auto valid_transpose_order = [expected_order](const std::shared_ptr<ngraph::Node>& node) -> bool {
        const auto transpose_pattern = ngraph::as_type_ptr<ngraph::opset1::Constant>(node);
        if (!transpose_pattern)
            return false;
        return transpose_pattern->cast_vector<int64_t>() == expected_order;
    };
    auto is_supported_transpose_tensor = [](const ngraph::descriptor::Tensor& t) {
        return is_supported_tensor(t) && ngraph::snippets::pass::TokenizeSnippets::supported_element_types.count(t.get_element_type()) != 0;
    };

    return node && node->get_output_target_inputs(0).size() == 1 && node->get_shape().size() == 4 &&
           valid_transpose_order(node->get_input_node_shared_ptr(1)) && is_supported_transpose_tensor(node->get_input_tensor(0));
}

auto tokenize_broadcast(const std::shared_ptr<ov::Node>& interm_op, ov::NodeVector& ordered_ops) -> void {
    // We can tokenize Broadcast op only when output shape of child doesn't depend on Broadcast shape without last dimension.
    // Snippets remove Broadcast op and insert BroadcastMove if last dimensions before and after Broadcast are different.
    // Otherwise, we can lose original shape.
    // Example:
    //        in0 [1, 1, 1]      in0 [1, 1, 1]              in0 [1, 1, 1]   in0 [1, 1, 1]
    //     Broadcast [1, 10, 1]    /                                 \       /
    //           \               /                --->>>                Add
    //                  Add                                              |
    //             Result [1, 10, 1]                              Result [1, 1, 1]

    ov::PartialShape new_output_shape(std::vector<ov::Dimension>{1});
    ov::NodeVector broadcast_nodes;

    auto skip_last_dim = [](const ov::PartialShape& shape) {
        return ov::PartialShape(std::vector<ov::Dimension>{shape.begin(), shape.end() - 1});
    };

    for (auto input : interm_op->inputs()) {
        auto broadcast = ov::as_type_ptr<ngraph::opset1::Broadcast>(input.get_source_output().get_node_shared_ptr());
        // TODO: Can we reuse AppropriateForSubgraph here? Seems like it's huge check for Broadcast
        if (broadcast && broadcast->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY &&
            broadcast->get_output_target_inputs(0).size() == 1) {
            broadcast_nodes.push_back(broadcast);

            const auto pshape = broadcast->get_input_partial_shape(0);
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ngraph::op::AutoBroadcastType::NUMPY);
            }
        } else {
            const auto pshape = input.get_partial_shape();
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ngraph::op::AutoBroadcastType::NUMPY);
            }
        }
    }

    if (!broadcast_nodes.empty()) {
        if (new_output_shape == skip_last_dim(interm_op->get_output_partial_shape(0))) {
            std::copy(broadcast_nodes.begin(), broadcast_nodes.end(), std::back_inserter(ordered_ops));
        }
    }
}

auto tokenize_reshape_around_softmax(std::shared_ptr<ov::Node>& interm_op,
                                     std::shared_ptr<ngraph::opset1::Reshape>& reshape,
                                     ngraph::NodeVector& ordered_ops) -> bool {
    reshape = ngraph::as_type_ptr<ngraph::opset1::Reshape>(interm_op);
    if (reshape) {
        const auto in_shape = reshape->get_input_shape(0);
        const auto out_shape = reshape->get_output_shape(0);
        if (in_shape.back() != out_shape.back() || reshape->get_output_target_inputs(0).size() != 1)
            return false;
        ordered_ops.push_back(reshape);
        interm_op = reshape->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
}

auto get_potential_body_params(const std::shared_ptr<ov::Node>& op) -> size_t {
    size_t count = 0;
    for (size_t i = 1; i < op->get_input_size(); ++i) {
        const auto input = op->input_value(i);
        const auto parent = input.get_node_shared_ptr();
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(parent);
        if (!(constant && (ngraph::shape_size(input.get_shape()) == 1 ||
                           ov::is_type<ov::op::v0::FakeQuantize>(op)||
                           ngraph::snippets::op::Subgraph::constant_input_should_be_inside_body(op)))) {
            count++;
        }
    }
    return count;
}

auto update_intermediate_supported_ops(std::shared_ptr<ov::Node>& interm_op, ngraph::NodeVector& ordered_ops,
                                       size_t& potential_body_params_count) -> bool {
    while (is_supported_intermediate_op(interm_op)) {
        // All supported intermediate ops have only one output port
        if (interm_op->get_output_target_inputs(0).size() != 1)
            return false;

        // Check for supported ops on branches: Broadcast/Elementwise (for example, dequantize ops)
        if (interm_op->get_input_size() > 1) {
            tokenize_broadcast(interm_op, ordered_ops);

            auto is_supported_branch_op = [&ordered_ops](const std::shared_ptr<ov::Node>& op) {
                return is_supported_intermediate_op(op) &&
                       ngraph::snippets::pass::GetSnippetsNodeType(op) != ngraph::snippets::pass::SnippetsNodeType::SkippedByPlugin &&
                       std::find(ordered_ops.begin(), ordered_ops.end(), op) == ordered_ops.end();
            };

            for (size_t i = 0; i < interm_op->get_input_size(); ++i) {
                const size_t shift = ordered_ops.size();
                auto parent = interm_op->get_input_node_shared_ptr(i);
                while (is_supported_branch_op(parent)) {
                    // All supported ops have only one output port
                    if (parent->get_output_target_inputs(0).size() != 1)
                        break;

                    // Add node only if there are scalar constants on inputs because of plugin-specific limitation
                    bool are_weights_scalar = true;
                    const auto parent_count = parent->get_input_size();
                    for (size_t i = 1; i < parent_count; ++i) {
                        are_weights_scalar = are_weights_scalar && ngraph::shape_size(parent->get_input_shape(i)) == 1;
                    }

                    ordered_ops.insert(ordered_ops.begin() + shift, parent);
                    // We think that sequence of ops goes through input port 0
                    // But can be Select here? If it can be, parent shouldn't be on input port 0. Need another way?
                    parent = parent->get_input_node_shared_ptr(0);
                }
            }

            potential_body_params_count += get_potential_body_params(interm_op);
        }

        ordered_ops.push_back(interm_op);
        interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
};
}  // namespace

ngraph::snippets::pass::TokenizeMHASnippets::TokenizeMHASnippets() {
    MATCHER_SCOPE(TokenizeMHASnippets);

    auto m_matmul0 = std::make_shared<ngraph::opset1::MatMul>(ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                              ngraph::pattern::any_input(ngraph::pattern::has_static_shape()));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_matmul0, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMHASnippets")
        auto& pattern_to_output = m.get_pattern_value_map();

        // Queries + Key + Values = 3 standard inputs of MHA
        size_t potential_body_params_count = 3;
        // After some transformations, a different number of Constants for some operations may be created
        // than the actual number of Constants during tokenization.
        // To avoid unsupported number of non-scalar Constants in the future (plugin specific limitation)
        // we should calculate potential number of non-scalar Constants that will be moved up from body.
        size_t hidden_virtual_ports_count = 0;
        // The count of potential unique Buffers - it's hidden virtual ports as well
        // We should go through Subgraph and calculate potential non-inplace Buffers count.
        // Example:
        //     Buffer - i32 [32, 128] -> ~ Loop ~ -> Buffer - i8 [32, 128]
        //     After each Loop iteration we should increment pointers of Buffers: accordingly on 4 byte and 1 byte for scalar case.
        //     It means that these Buffers cannot be inplace => Each Buffer should have the own register
        // For that we can just check the following "branches":
        //  - Between MatMul0 and MatMul1 - Softmax is sync point. The operations between MatMul0 -> Softmax and Softmax -> MatMul1
        //                                  will be fused into one loop aftet conversion to snippet dialect (Becase it's just FQ, Eltwise nodes)
        //  - Between MatMul0 and Transpose1 - At the moment operations after Transpose1 cannot be fused in Transpose Loop (to avoid performance regressions).
        //                                     But operations after Transpose1 and before MatMul0  will be fused into one loop as well (look at first point)
        // Note: If the pass is updated, need to check the new possible branches for potential non-inplace Buffers!
        // Default value is 1 because
        //  - Firstly Softmax always need to have Buffers
        //  - Secondly Softmax need 2 Buffer but they can be inplace - One virtual port is enough for Softmax
        size_t buffer_count = 1;
        std::string fused_names;
        ngraph::NodeVector ordered_ops;

        /* ======== Matcher Pass ========== */

        /****** Skeleton ******/
        /* Skeleton on MHA-pattern is:
         *              \     /
         *              MatMul0
         *                 |
         *    Eltwise/Select/Reshape/FakeQuantize
         *                 |
         *              Softmax
         *                 |
         *    Eltwise/Select/Reshape/FakeQuantize
         *                  \      /
         *                   MatMul1
         */
        const auto matmul0 = ngraph::as_type_ptr<ngraph::opset1::MatMul>(pattern_to_output.at(m_matmul0).get_node_shared_ptr());
        if (!matmul0 || matmul0->get_output_target_inputs(0).size() != 1 || matmul0->get_transpose_a() ||
            !is_supported_tensor(matmul0->get_input_tensor(0)) || !is_supported_tensor(matmul0->get_input_tensor(1)))
            return false;

        const auto matmul0_prc = op::Brgemm::get_output_type(matmul0->get_input_element_type(0), matmul0->get_input_element_type(1));
        if (matmul0_prc == element::undefined) {
            return false;
        }

        // Between MatMul0 and Softmax will be the one Loop because of LoopFusing optimization.
        // The Loop will have one Buffer with the same shape both on input and output.
        // Need to check for precision to get if we need one more register for Buffer
        if (matmul0_prc.size() != ov::element::f32.size()) {
            if (buffer_count < 2)
                buffer_count++;
        }

        ordered_ops.push_back(matmul0);

        auto interm_op = matmul0->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        // Add supported operations which are between MatMul0 and Softmax to ordered_ops
        if (!update_intermediate_supported_ops(interm_op, ordered_ops, potential_body_params_count))
            return false;

        std::shared_ptr<ngraph::opset1::Reshape> reshape0 = nullptr;
        if (!tokenize_reshape_around_softmax(interm_op, reshape0, ordered_ops))
            return false;

        int64_t axis = 0;
        const auto rank = interm_op->get_input_partial_shape(0).rank();
        if (const auto softmax_v8 = ngraph::as_type_ptr<ngraph::opset8::Softmax>(interm_op)) {
            axis = ngraph::normalize_axis(interm_op->get_friendly_name(), softmax_v8->get_axis(), rank);
        } else if (const auto softmax_v1 = ngraph::as_type_ptr<ngraph::opset1::Softmax>(interm_op)) {
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }

        if (axis != rank.get_length() - 1 || interm_op->get_output_target_inputs(0).size() != 1)
            return false;
        ordered_ops.push_back(interm_op);

        interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        std::shared_ptr<ngraph::opset1::Reshape> reshape1 = nullptr;
        if (!tokenize_reshape_around_softmax(interm_op, reshape1, ordered_ops))
            return false;

        if (((reshape0 == nullptr) != (reshape1 == nullptr)) ||
             (reshape0 && reshape1 && (reshape0->get_input_shape(0) != reshape1->get_output_shape(0))))
            return false;

        // Add supported operations which are between Softmax and MatMul1 to ordered_ops
        if (!update_intermediate_supported_ops(interm_op, ordered_ops, potential_body_params_count))
            return false;

        const auto matmul1 = ngraph::as_type_ptr<ngraph::opset1::MatMul>(interm_op);
        if (!matmul1 || matmul1->get_output_target_inputs(0).size() != 1 || matmul1->get_transpose_a() || matmul1->get_transpose_b() ||
            op::Brgemm::get_output_type(matmul1->get_input_element_type(0), matmul1->get_input_element_type(1)) == element::undefined ||
            !is_supported_tensor(matmul1->get_input_tensor(0)) || !is_supported_tensor(matmul1->get_input_tensor(1)))
            return false;

        if (transformation_callback(matmul1)) {
            return false;
        }

        // Between Softmax and MatMul1 will be the one Loop because of LoopFusing optimization.
        // The Loop will have one Buffer with the same shape both on input and output.
        // Need to check for precision to get if we need one more register for Buffer
        if (matmul1->get_input_element_type(0).size() != ov::element::f32.size()) {
            buffer_count++;
        }

        /***********************/

        /***** Transposes *****/
        /* There may be Transpose and Reshape ops on inputs and outputs of MHA-pattern skeleton
         * We can add them into Subgraph body
         */

        // First input branch of MatMul0 should be executed before second input branch of MatMul0,
        // so firstly we insert Transpose1 on the beginning of ordered_ops and then Transpose1
        bool are_weights_scalar = true;
        auto parent = matmul0->get_input_node_shared_ptr(1);
        while (is_supported_intermediate_op(parent)) {
            // All supported ops have only one output port
            if (parent->get_output_target_inputs(0).size() != 1)
                break;
            const auto parent_count = parent->get_input_size();
            for (size_t i = 1; i < parent_count; ++i) {
                are_weights_scalar = are_weights_scalar && ngraph::shape_size(parent->get_input_shape(i)) == 1;
            }
            potential_body_params_count += get_potential_body_params(parent);
            ordered_ops.insert(ordered_ops.begin(), parent);
            // [107731] To go always through 0-th port - is it safe?
            parent = parent->get_input_node_shared_ptr(0);
        }

        auto transpose1 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent);
        if (matmul0->get_transpose_b()) {
            if (is_valid_transpose(transpose1, {0, 2, 1, 3})) {
                // We can support several ops between MatMul0 with transposed_b and Transpose1 with 0213 order
                // only if these ops have scalar shapes on other inputs.
                // There is transformation ExplicitTransposeMatMulInputs that set supported order and transposed_b(false).
                // We can allow to call this pass only if ops have scalar shapes to avoid shape mismatching
                if (are_weights_scalar) {
                    ordered_ops.insert(ordered_ops.begin(), transpose1);
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            if (is_valid_transpose(transpose1, {0, 2, 3, 1})) {
                ordered_ops.insert(ordered_ops.begin(), transpose1);
            }
        }

        if (transpose1) {
            // Between Transpose1 and MatMul0 will be the one Loop because of LoopFusing optimization.
            // The Loop will have one Buffer with the same shape both on input and output.
            // Need to check for precision to get if we need one more register for Buffer
            if (matmul0->get_input_element_type(1).size() != transpose1->get_output_element_type(0).size()) {
                buffer_count++;
            }
        }

        const auto transpose0 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(matmul0->get_input_node_shared_ptr(0));
        if (is_valid_transpose(transpose0, {0, 2, 1, 3})) {
            ordered_ops.insert(ordered_ops.begin(), transpose0);
        } else if (matmul0->get_transpose_b()) {
            return false;
        }

        const auto transpose2 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(matmul1->get_input_node_shared_ptr(1));
        if (is_valid_transpose(transpose2, {0, 2, 1, 3})) {
            ordered_ops.push_back(transpose2);
        }
        ordered_ops.push_back(matmul1);

        bool are_ops_after_matmul2 = false;
        auto child = matmul1->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        while (is_supported_intermediate_op(child)) {
            are_ops_after_matmul2 = true;
            // All supported ops have only one output port
            if (child->get_output_target_inputs(0).size() != 1)
                break;

            potential_body_params_count += get_potential_body_params(child);
            // [75567]: move this plugin-specific constraint to the plugin callback
            //          We cannot collapse op to Subgraph if count of potential Parameter and Result count is higher 12
            if (potential_body_params_count + child->get_output_target_inputs(0).size() + hidden_virtual_ports_count + buffer_count > 12) {
                break;
            }

            ordered_ops.push_back(child);
            child = child->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        }

        // At the moment Snippets don't support nodes between MatMul2 and Transpose3 due to Loop and strided calculations limitations
        //     MatMul2
        //  <Supported ops>
        //    Transpose3
        // TODO: Add check for precision of MatMul (we cannot collapse Transpose to I8/BF16 MatMul)
        if (!are_ops_after_matmul2) {
            auto transpose3 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(child);
            if (is_valid_transpose(transpose3, {0, 2, 1, 3})) {
                ordered_ops.push_back(transpose3);
            }
        }

        /**********************/

        /* ================================ */

        /* ====== Subgraph creation ======= */

        // [75567]: move this plugin-specific constraint to the plugin callback
        const auto last_node = ordered_ops.back();
        if (potential_body_params_count + last_node->get_output_size() + hidden_virtual_ports_count + buffer_count > 12) {
            return false;
        }

        ngraph::OutputVector body_inputs, subgraph_inputs;
        ngraph::ParameterVector body_parameters;
        ngraph::ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        auto create_body_inputs = [&](const std::shared_ptr<ngraph::Node>& node) -> void {
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                const auto input = node->input(i);
                const auto parent = input.get_source_output().get_node_shared_ptr();
                const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(parent);
                if (constant && (ngraph::shape_size(input.get_shape()) == 1 ||
                                 ov::is_type<ov::op::v0::FakeQuantize>(node) ||
                                 op::Subgraph::constant_input_should_be_inside_body(node))) {
                    // If Constant has one consumer - target node, we add Constant to body_inputs
                    // If Constant has several consumers, we should check that all these consumers are inside Subgraph body
                    // and if all of them are inside body, we can explicitly add Constant to the body_inputs, otherwise we should
                    // make a copy and add copy of Constant to body_inputs
                    // For example, this case is especially valid for Transposes nodes
                    //              (several Transposes have the same order so there can be the common Constant with this order)
                    if (constant->get_output_target_inputs(0).size() == 1) {
                        body_inputs.push_back(input.get_source_output());
                    } else {
                        const auto constant_consumers = constant->get_output_target_inputs(0);
                        bool all_consumers_are_inside = std::all_of(constant_consumers.begin(), constant_consumers.end(),
                                                                    [&ordered_ops](const ngraph::Input<ngraph::Node>& input) {
                                                                        return std::find(ordered_ops.begin(), ordered_ops.end(),
                                                                                         input.get_node()->shared_from_this()) != ordered_ops.end();
                                                                    });
                        if (all_consumers_are_inside) {
                            body_inputs.push_back(input.get_source_output());
                        } else {
                            const auto constant_copy = constant->clone_with_new_inputs({});
                            node->set_argument(input.get_index(), constant_copy);
                            body_inputs.push_back(constant_copy);
                        }
                    }
                } else if (std::find(ordered_ops.begin(), ordered_ops.end(), parent) == ordered_ops.end()) {
                    auto parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                    body_parameters.push_back(parameter);
                    body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
                    body_inputs.push_back(parameter->output(0));

                    subgraph_inputs.push_back(input.get_source_output());

                    node->input(i).replace_source_output(parameter);
                }
            }
        };

        for (const auto& op : ordered_ops) {
            create_body_inputs(op);
            op->clear_control_dependencies();
            fused_names += op->get_friendly_name() + ",";
        }

        for (const auto& output : last_node->outputs()) {
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }
        for (const auto& output : last_node->outputs()) {
            body_results.push_back(std::make_shared<ngraph::opset1::Result>(last_node->output(output.get_index())));
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            throw ngraph_error("body results and node results size mismatch during subgraph collapse");
        }

        auto body = op::create_body(last_node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = std::make_shared<op::Subgraph>(subgraph_inputs, body);
        // Copy runtime info from last node to subgraph - to copy topological order
        copy_runtime_info(last_node, subgraph);
        subgraph->set_friendly_name(last_node->get_friendly_name());

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (const auto& target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        op::update_out_tensor_name(subgraph);

        subgraph->validate_and_infer_types();

        auto act_body = subgraph->body_ptr();
        for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
            act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        subgraph->get_rt_info()["originalLayersNames"] = fused_names;
        subgraph->set_virtual_port_count(hidden_virtual_ports_count);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into the MHA Subgraph in common Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);

        return true;

        /* ================================ */
    });
}
