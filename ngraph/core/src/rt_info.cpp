// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/rt_info.hpp"
#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

ngraph::Node::RTMap mergeRuntimeInfo(const ngraph::NodeVector& nodes)
{
    ngraph::Node::RTMap mergedInfo;
    for (auto& node : nodes)
    {
        for (auto& item : node->get_rt_info())
        {
            mergedInfo[item.first] = item.second;
        }
    }

    ngraph::Node::RTMap newInfo;
    for (auto& item : mergedInfo)
    {
        size_t attributes_count = 0;
        for (auto& node : nodes)
        {
            const auto& rt_info = node->get_rt_info();
            if (rt_info.count(item.first))
            {
                attributes_count++;
            }
        }

        if (attributes_count == 1)
        {
            newInfo[item.first] = item.second;
        }
        else if (auto merge_attr = item.second->merge(nodes))
        {
            newInfo[item.first] = merge_attr;
        }
    }

    return newInfo;
}

// TODO: workaround to copy (not merge) attributes for the same types
void copy_runtime_info_for_ports(const std::shared_ptr<ngraph::Node>& from,
                                 const std::shared_ptr<ngraph::Node>& to)
{
    if (to->get_type_info() != from->get_type_info())
    {
        return;
    }

    for (size_t i = 0; i < from->get_input_size(); ++i)
    {
        auto& source = from->input(i).get_rt_info();
        auto& target = to->input(i).get_rt_info();
        for (auto attribute : source)
        {
            target[attribute.first] = attribute.second;
        }
    }
    for (size_t i = 0; i < from->get_output_size(); ++i)
    {
        auto& source = from->output(i).get_rt_info();
        auto& target = to->output(i).get_rt_info();
        for (auto attribute : source)
        {
            target[attribute.first] = attribute.second;
        }
    }
}

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to)
{
    auto& rtInfoFrom = from->get_rt_info();
    auto& rtInfoTo = to->get_rt_info();
    rtInfoTo = rtInfoFrom;
    copy_runtime_info_for_ports(from, to);
}

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to)
{
    for (auto& op : to)
    {
        copy_runtime_info(from, op);
        copy_runtime_info_for_ports(from, op);
    }
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to)
{
    auto& rtInfoTo = to->get_rt_info();
    rtInfoTo = mergeRuntimeInfo(from);

    for (auto& fromNode : from)
    {
        copy_runtime_info_for_ports(fromNode, to);
    }
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to)
{
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to)
    {
        auto& rtInfoTo = node->get_rt_info();
        rtInfoTo = mergedInfo;
    }

    for (auto& fromNode : from)
    {
        for (auto& toNode : to)
        {
            copy_runtime_info_for_ports(fromNode, toNode);
        }
    }
}
