// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <queue>
#include <string>
#include <transformations/smart_reshape/interpolate_concat.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using Attrs = ngraph::opset8::Interpolate::InterpolateAttrs;
using ShapeCalcMode = ngraph::opset8::Interpolate::ShapeCalcMode;
using InterpolateMode = ngraph::opset8::Interpolate::InterpolateMode;
using CoordinateTransformMode = ngraph::opset8::Interpolate::CoordinateTransformMode;
using NearestMode = ngraph::opset8::Interpolate::NearestMode;

TEST_F(TransformationTestsF, InterpolateConcatReplacementReplacement_case1) {
    {
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{1,3,30,40});
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{1,4,60,160});
        constexpr auto elem_count = 236;
        constexpr auto target_shape_elem_type = element::i64;
        Attrs attrs = Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL, NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        auto sizes_const = ngraph::opset8::Constant::create(target_shape_elem_type, {2}, {60, 160});
        auto scales_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3.0f});
        auto axes_const = ngraph::opset8::Constant::create(target_shape_elem_type, {2}, {2, 3});
        auto interpolate_node = std::make_shared<ngraph::opset8::Interpolate>(input1,
                                                                             sizes_const,
                                                                             scales_const,
                                                                             axes_const,
                                                                             attrs);
        auto concat_node = std::make_shared<ngraph::opset8::Concat>(ov::OutputVector{interpolate_node, input2}, 1);

        function = std::make_shared<Function>(OutputVector{concat_node}, ParameterVector{input1, input2});

        manager.register_pass<ov::pass::InterpolateConcatReplacement>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{1,3,30,40});
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{1,4,60,160});
        constexpr auto elem_count = 236;
        constexpr auto target_shape_elem_type = element::i64;
        Attrs attrs = Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL, NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};

        std::vector<int32_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        //auto sizes_const = ngraph::opset8::Constant::create(target_shape_elem_type, {2}, {60, 160});
        auto scales_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3.0f});
        auto axes_const = ngraph::opset8::Constant::create(target_shape_elem_type, {2}, {2, 3});
    

        auto shape_node = std::make_shared<opset8::ShapeOf>(input2);

        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_indices_node = opset8::Constant::create(element::i64, {2}, {2, 3});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, gather_indices_node, gather_axis_node);


        auto interpolate_node = std::make_shared<ngraph::opset8::Interpolate>(input1,
                                                                             gather_node,
                                                                             scales_const,
                                                                             axes_const,
                                                                             attrs);
        auto concat_node = std::make_shared<ngraph::opset8::Concat>(ov::OutputVector{interpolate_node, input2}, 1);

        function_ref = std::make_shared<Function>(OutputVector{concat_node}, ParameterVector{input1, input2});
    }
}