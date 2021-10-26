// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/function.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Subgraph
 * @brief An operation that is implemented by a function
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Subgraph : public ngraph::op::Op {
public:
    OPENVINO_OP("Subgraph", "SnippetsOpset");

    // < 1, 42, 17, 15, 16> < 0, 1, 2, 3, 1>
    // should be:
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15, 16> < 0, 1, 2, 3, 1>
    // D = < 1, 42,  1, 1 > -> < 1, 3,  1,  1, 16> < 0, 1, 2, 3, 1> ???
    // C = A + B
    // C = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    //
    // how it works now (multi-demention broadcast):
    // [BroadcastLoad] doesn't perform post increment
    // [Load] performs += vlan
    // [ScalarLoad] performs += 1
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15,  1> < 0, 1, 2, 3, 1>
    // [A]     [B]
    // [Load]  [ScalarLoad] <- should consider AxisVector to choose right type of load
    //         [Broadcast]
    //   [Add]
    //  [Store]
    //    [C]
    // C = A + B
    // C = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    //
    // Multiple-dimension broadcasts support?
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 4>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15,  1> < 0, 1, 2, 3, 4>
    //
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 4>
    // B = < 1,  1, 17, 15> -> < 1, 3, 17, 15,  1> < 0, 1, 2, 3, 4>
    //
    // Collapse moat varying dimensions with broadcast
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 3, 17, 15,  1> < 0, 1, 2, 3, 1>
    //
    // Collapse for mixed broadcast
    // A = < 1, 3, 17, 15, 32> < 0, 1, 2, 3, 4>
    // B = < 1, 3, 17,  1, 32> < 0, 1, 2, 3, 4>
    // C = < 1, 3,  1, 15, 32> < 0, 1, 2, 3, 4>
    //
    // D = < 1, 3, 17, 15, 32> < 0, 1, 2, 3, 4>
    // E = < 1, 3, 17,  1, 32> < 0, 1, 2, 3, 4>
    using BlockedShape = std::tuple<ngraph::Shape, ngraph::AxisVector, ngraph::element::Type>;
    using BlockedShapeVector = std::vector<BlockedShape>;

    Subgraph(const OutputVector& args, std::shared_ptr<Function> body);

    Subgraph(const NodeVector& args, std::shared_ptr<Function> body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<Function> get_body() const {
        return m_body;
    }

    std::shared_ptr<ngraph::snippets::Generator> get_generator() const {
        return m_generator;
    }

    std::shared_ptr<Subgraph> make_canonical_from_this();

    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes,
                                ngraph::pass::Manager opt = ngraph::pass::Manager());

    /// Set a new body for the op; body needs to satisfy requirements on inputs/outputs
    void set_body(std::shared_ptr<Function> body);

    // plugin sets generator for a snippet to some specific generator.
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<ngraph::snippets::Generator> generator);

    void print() const;
    void print_statistics(bool verbose);

    void serialize() const;

    static auto wrap_node_as_subgraph(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<Subgraph>;

private:
    void canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);
    void convert_to_snippet_dialect();

    std::shared_ptr<Function> m_body;
    std::shared_ptr<ngraph::snippets::Generator> m_generator;
};

static inline std::ostream& operator<<(std::ostream& os, const op::Subgraph::BlockedShape& blocked_shape) {
    os << std::get<0>(blocked_shape) << " " << std::get<1>(blocked_shape) << " " << std::get<2>(blocked_shape);
    return os;
}

static inline auto is_scalar_constant(const std::shared_ptr<ngraph::Node>& source_output_node) -> bool {
    return ngraph::is_type<ngraph::opset1::Constant>(source_output_node) &&
        (source_output_node->get_shape().empty() || ngraph::shape_size(source_output_node->get_shape()) == 1);
};

static inline auto create_body(std::string name, const ngraph::ResultVector& results, const ngraph::ParameterVector& parameters) ->
    std::shared_ptr<ngraph::Function> {
    auto body = std::make_shared<ngraph::Function>(results, parameters, name);
    return body;
};

static inline auto build_subgraph(const std::shared_ptr<ngraph::Node>& node, const ngraph::OutputVector& inputs,
                                  const std::shared_ptr<ngraph::Function>& body, const std::string name = "")
    -> std::shared_ptr<Subgraph>{
    auto subgraph = std::make_shared<Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(name.empty() ? node->get_friendly_name() : name);
    return subgraph;
};

}  // namespace op
}  // namespace snippets
}  // namespace ngraph
