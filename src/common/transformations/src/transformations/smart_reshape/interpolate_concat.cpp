// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <transformations/smart_reshape/interpolate_concat.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::InterpolateConcatReplacement::InterpolateConcatReplacement() {
    MATCHER_SCOPE(InterpolateConcatReplacement);
    auto concat_pattern_node = pattern::wrap_type<ngraph::opset8::Concat>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto concat = m.get_match_root();
        std::cout<<"concat in InterpolateConcatReplacement" << std::endl;
        std::cout<<"" << std::endl;

        std::vector<std::shared_ptr<ngraph::Node>> interpolate_nodes;
        std::vector<std::shared_ptr<ngraph::Node>> non_interpolate_nodes;
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto input_op = concat->get_input_node_shared_ptr(i);
            const auto interpolate_ptr = std::dynamic_pointer_cast<ngraph::opset8::Interpolate>(input_op);
            if (interpolate_ptr) {
                interpolate_nodes.push_back(input_op);
            }
            else {
                non_interpolate_nodes.push_back(input_op);
            }
        }
        if (interpolate_nodes.size() != 1 || non_interpolate_nodes.size() != 1) {
            return false;
        }
        auto interpolate_node = interpolate_nodes[0];
        auto concat_input_node = non_interpolate_nodes[0];
        auto sizes_node = interpolate_node->get_input_node_shared_ptr(1);
        if (!std::dynamic_pointer_cast<ngraph::opset8::Constant>(sizes_node)) {
            return false;
        }

        auto axes_node = interpolate_node->get_input_node_shared_ptr(3);
        const auto axes_node_ptr = std::dynamic_pointer_cast<ngraph::opset8::Constant>(axes_node);
        if (!axes_node_ptr)
            return false;
        if (!axes_node_ptr->get_element_type().is_integral())
            return false;

        auto indices_shape = axes_node_ptr->get_shape();
        const auto& indices = axes_node_ptr->cast_vector<int64_t>();


        auto shape_node = std::make_shared<opset8::ShapeOf>(concat_input_node);

        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_indices_node = opset8::Constant::create(element::i64, indices_shape, indices);
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, gather_indices_node, gather_axis_node);
        
        copy_runtime_info(sizes_node, {gather_node, shape_node});
        ov::replace_node(sizes_node, gather_node);
        

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern_node, matcher_name);
    this->register_matcher(m, callback);
}
