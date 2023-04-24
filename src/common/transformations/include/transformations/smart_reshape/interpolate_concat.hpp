// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API InterpolateConcatReplacement;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief InterpolateConcatReplacement makes Interpolate with Concat reshapeble
 */

class ov::pass::InterpolateConcatReplacement : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InterpolateConcatReplacement", "0");
    InterpolateConcatReplacement();
};
