// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_horizontal_fusing.hpp"
#include "transformations/utils/utils.hpp"

#include "itt.hpp"
#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MatMulHorizontalFusion, "MatMulHorizontalFusion", 0);

bool ngraph::pass::MatMulHorizontalFusion::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(MatMulHorizontalFusion);
    auto is_matmul_with_weights = [](const std::shared_ptr<ngraph::Node>& node) {
        auto weights = node->get_input_node_shared_ptr(1);
        if (ngraph::is_type<ngraph::opset8::Constant>(weights)) {
            return true;
        }

        if (ngraph::is_type<ngraph::opset8::Multiply>(weights) &&
            ngraph::is_type<ngraph::opset8::Constant>(weights->get_input_node_shared_ptr(1))) {
            weights = weights->get_input_node_shared_ptr(0);
        }

        if (ngraph::is_type<ngraph::opset8::Subtract>(weights)) {
            auto constant_path = weights->get_input_node_shared_ptr(1);
            if ((ngraph::is_type<ngraph::opset8::Constant>(constant_path)) ||
                (ngraph::is_type<ngraph::opset8::Convert>(constant_path) &&
                 ngraph::is_type<ngraph::opset8::Constant>(constant_path->get_input_node_shared_ptr(0)))) {
                weights = weights->get_input_node_shared_ptr(0);
            }
        }

        if (ngraph::is_type<ngraph::opset8::Convert>(weights)) {
            weights = weights->get_input_node_shared_ptr(0);
            return ngraph::is_type<ngraph::opset8::Constant>(weights);
        }

        return false;
    };

    auto fuse_matmuls = [](const ngraph::NodeVector& matmuls, const ngraph::NodeVector& add_nodes) {
        const auto matmul = ngraph::as_type_ptr<ngraph::opset8::MatMul>(matmuls[0]);
        const size_t matmuls_num = matmuls.size();
        if (matmuls_num < 2ul) {
            return false;
        }

        ngraph::NodeVector biases;
        biases.reserve(matmuls_num);
        bool fuse_biases = add_nodes.size() == matmuls_num;
        if (fuse_biases) {
            // we already checked that bias is a constant and it is possible to use its shape
            ngraph::Shape bias_shape = add_nodes[0]->get_input_shape(1);
            for (const auto& add_node : add_nodes) {
                if (add_node->get_input_shape(1) != bias_shape) {
                    fuse_biases = false;
                    break;
                }
                biases.emplace_back(add_node->get_input_node_shared_ptr(1));
            }
        }

        auto fuse_weights_path = [](const ngraph::NodeVector& matmuls, const bool transpose_weights) {
            ngraph::NodeVector mul_constants;
            ngraph::NodeVector sub_constants;
            std::shared_ptr<ngraph::Node> convert;
            ngraph::NodeVector weights;

            for (const auto& elem : matmuls) {
                auto weights_path = elem->get_input_node_shared_ptr(1);
                if (ngraph::is_type<ngraph::opset8::Multiply>(weights_path)) {
                    mul_constants.emplace_back(weights_path->get_input_node_shared_ptr(1));
                    weights_path = weights_path->get_input_node_shared_ptr(0);
                }
                if (ngraph::is_type<ngraph::opset8::Subtract>(weights_path)) {
                    sub_constants.emplace_back(weights_path->get_input_node_shared_ptr(1));
                    weights_path = weights_path->get_input_node_shared_ptr(0);
                }
                if (ngraph::is_type<ngraph::opset8::Convert>(weights_path)) {
                    convert = weights_path;
                    weights_path = weights_path->get_input_node_shared_ptr(0);
                }
                if (ngraph::is_type<ngraph::opset8::Constant>(weights_path)) {
                    weights.emplace_back(weights_path);
                }
            }

            const auto matmul = matmuls[0];
            const auto weights_shape = matmul->get_input_shape(1);
            const size_t concat_axis = transpose_weights ? weights_shape.size() - 2 : weights_shape.size() - 1;
            std::shared_ptr<ngraph::Node> new_weights = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(weights, concat_axis);
            ngraph::copy_runtime_info(weights, new_weights);

            if (convert) {
                new_weights = convert->clone_with_new_inputs({ new_weights });
                ngraph::copy_runtime_info(convert, new_weights);
            }

            if (!sub_constants.empty()) {
                const auto new_sub_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(sub_constants, concat_axis);
                new_weights = std::make_shared<ngraph::opset8::Subtract>(new_weights, new_sub_const);
                ngraph::copy_runtime_info(sub_constants[0]->output(0).target_inputs()[0], new_weights);
            }

            if (!mul_constants.empty()) {
                const auto new_mul_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(mul_constants, concat_axis);
                new_weights = std::make_shared<ngraph::opset8::Multiply>(new_weights, new_mul_const);
                ngraph::copy_runtime_info(mul_constants[0]->output(0).target_inputs()[0], new_weights);
            }

            return new_weights;
        };

        const auto new_weights = fuse_weights_path(matmuls, matmul->get_transpose_b());
        const auto new_matmul = matmul->clone_with_new_inputs({ matmul->input_value(0), new_weights });
        new_matmul->set_friendly_name(matmul->get_friendly_name() + "/Fused");
        ngraph::copy_runtime_info(matmuls, new_matmul);

        std::shared_ptr<ngraph::Node> last_fused_node = new_matmul;
        if (fuse_biases) {
            std::int64_t biases_concat_axis = 0;
            const auto bias_shape = biases[0]->get_output_shape(0);
            for (size_t i = 0; i < bias_shape.size(); ++i) {
                if (bias_shape[i] > 1ul) {
                    biases_concat_axis = i;
                    break;
                }
            }

            const auto new_biases = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(biases, biases_concat_axis);
            last_fused_node = std::make_shared<ngraph::opset8::Add>(last_fused_node, new_biases);
            ngraph::copy_runtime_info(biases, last_fused_node);
        }

        const auto matmul_out_rank = matmul->get_output_partial_shape(0).rank().get_length();
        const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { matmul_out_rank - 1 });
        const auto split = std::make_shared<ngraph::opset8::Split>(last_fused_node, split_axis, matmuls_num);
        ngraph::copy_runtime_info(last_fused_node, split);

        const auto& last_original_nodes = fuse_biases ? add_nodes : matmuls;
        for (size_t i = 0; i < matmuls_num; ++i) {
            auto output = last_original_nodes[i]->output(0);
            output.replace(split->output(i));
        }

        return true;
    };

    bool rewritten = false;
    for (const auto& node : f->get_ordered_ops()) {
        const auto outputs = node->outputs();
        if (std::all_of(outputs.begin(), outputs.end(), [](const ngraph::Output<Node>& elem) { return elem.target_inputs().size() == 1ul; })) {
            continue;
        }

        for (const auto& output : outputs) {
            const auto consumers = output.target_inputs();
            std::vector<std::shared_ptr<ngraph::opset8::MatMul>> matmuls;
            for (const auto& consumer : consumers) {
                const auto matmul = ngraph::as_type_ptr<ngraph::opset8::MatMul>(consumer);
                if (matmul && matmul->get_output_partial_shape(0).rank().is_static() && is_matmul_with_weights(matmul)) {
                    matmuls.emplace_back(matmul);
                }
            }

            if (matmuls.size() < 2) {
                continue;
            }

            auto get_weights = [](const std::shared_ptr<ngraph::Node>& matmul) {
                auto weighs_path = matmul->get_input_node_shared_ptr(1);
                if (ngraph::is_type<ngraph::opset8::Constant>(weighs_path)) {
                    return weighs_path;
                }

                if (ngraph::is_type<ngraph::opset8::Multiply>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                if (ngraph::is_type<ngraph::opset8::Subtract>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                if (ngraph::is_type<ngraph::opset8::Convert>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                return weighs_path;
            };

            const auto matmul = matmuls[0];
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            const auto weights = get_weights(matmul);

            auto gold_and_cur_are_similar = [&](const std::shared_ptr<ngraph::opset8::MatMul>& target) {
                if (target->get_transpose_a() != transpose_a ||
                    target->get_transpose_b() != transpose_b) {
                    return false;
                }

                auto target_weights = get_weights(target);
                if (weights->get_output_element_type(0) != target_weights->get_output_element_type(0) ||
                    weights->get_output_shape(0) != target_weights->get_output_shape(0)) {
                    return false;
                }

                return true;
            };

            NodeVector matmuls_to_fuse;
            NodeVector biases_to_fuse;
            for (size_t i = 0; i < matmuls.size(); ++i) {
                const auto neighbor_matmul_consumers = matmuls[i]->output(0).target_inputs();
                if (std::any_of(neighbor_matmul_consumers.begin(), neighbor_matmul_consumers.end(),
                    [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
                    continue;
                }

                if (!gold_and_cur_are_similar(matmuls[i])) {
                    continue;
                }

                matmuls_to_fuse.emplace_back(matmuls[i]);
                if (neighbor_matmul_consumers.size() == 1ul) {
                    const auto bias_node = neighbor_matmul_consumers[0];
                    if (ngraph::is_type<ngraph::opset8::Add>(bias_node) &&
                        ngraph::is_type<ngraph::opset8::Constant>(bias_node->get_input_node_shared_ptr(1))) {
                        const auto bias_consumers = bias_node->output(0).target_inputs();

                        if (std::all_of(bias_consumers.begin(), bias_consumers.end(),
                            [](const std::shared_ptr<ngraph::Node>& elem) { return !ngraph::is_type<ngraph::opset8::Result>(elem); })) {
                            biases_to_fuse.emplace_back(bias_node);
                        }
                    }
                }
            }

            rewritten |= fuse_matmuls(matmuls_to_fuse, biases_to_fuse);
        }
    }

    return rewritten;
}
