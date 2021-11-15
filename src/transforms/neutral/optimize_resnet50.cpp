/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/optimize_resnet50.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

std::vector<float> ToFloats(const std::vector<std::byte> &bytes)
{
    std::vector<float> floats(bytes.size() / sizeof(float), 0.0f);
    std::copy_n(bytes.begin(), floats.size() * sizeof(float),
        std::as_writable_bytes(std::span(floats)).begin());
    return floats;
}

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_mul_add_to_dw_transform::on_try_match(node &node, transform_context &context)
{
    // constant *mul_const, *add_const;
    if (auto b_mul = node_cast<binary>(node))
    {
        if (b_mul->binary_op() == binary_mul)
        {
            if (try_get_direct_parent<conv2d>(*b_mul, 0) || try_get_direct_parent<conv2d>(*b_mul, 1))
                return false;
            if (auto tmp_binary = try_get_direct_parent<binary>(*b_mul, 0))
            {
                if (tmp_binary->binary_op() == binary_max)
                    return false;
            }
            if (auto b_add = try_get_direct_child<binary>(*b_mul))
            {
                if (b_add->output().connections().size() == 1 && b_add->output().connections()[0]->owner().runtime_opcode() == op_conv2d)
                {
                    if (b_add->binary_op() == binary_add && b_add->fused_activation().min == 0.f && b_add->fused_activation().max == std::numeric_limits<float>::infinity())
                    {
                        if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 1))
                        {
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.matched_nodes.emplace_back(mul_const);
                        }
                        else if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 0))
                        {
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.matched_nodes.emplace_back(mul_const);
                        }
                        else
                        {
                            return false;
                        }
                        if (auto add_const = try_get_direct_parent<constant>(*b_add, 1))
                        {
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.matched_nodes.emplace_back(add_const);
                        }
                        else if (auto add_const = try_get_direct_parent<constant>(*b_add, 0))
                        {
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.matched_nodes.emplace_back(add_const);
                        }
                        else
                        {
                            return false;
                        }

                        context.outputs.emplace_back(&b_add->output());
                        context.matched_nodes.emplace_back(b_mul);
                        context.matched_nodes.emplace_back(b_add);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void fold_mul_add_to_dw_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &const_mul = static_cast<constant &>(*context.matched_nodes[0]);
    auto &const_add = static_cast<constant &>(*context.matched_nodes[1]);
    // auto &b_mul = static_cast<binary &>(*context.matched_nodes[2]);
    auto &b_add = static_cast<binary &>(*context.matched_nodes[3]);

    std::vector<std::byte> mul_const_byte { const_mul.data().begin(), const_mul.data().end() };
    std::vector<std::byte> add_const_byte { const_add.data().begin(), const_add.data().end() };
    auto mul_const = ToFloats(mul_const_byte);
    auto add_const = ToFloats(add_const_byte);
    std::vector<float> max_const(output.shape()[1]);
    for (size_t i = 0; i < output.shape()[1]; i++)
    {
        max_const[i] = (0.f - static_cast<float>(add_const[i])) / static_cast<float>(mul_const[i]);
    }
    auto const_max = context.graph.emplace<constant>(dt_float32, shape_t { max_const.size(), 1, 1 }, max_const);
    auto b_max = context.graph.emplace<binary>(binary_max, output.shape(), const_max->output().shape(), value_range<float>::full());
    const_max->name(b_add.name() + "_relu2max_const");
    b_max->name(b_add.name() + "_relu2max");
    b_max->input_a().connect(output);
    b_max->input_b().connect(const_max->output());
    // b_mul  input
    context.inputs[0]->connect(b_max->output());
    b_add.fused_activation() = value_range<float>::full();
}

// bool fold_mul_add_to_dw_transform::on_try_match(node &node, transform_context &context)
// {
//     // constant *mul_const, *add_const;
//     if (auto b_mul = node_cast<binary>(node))
//     {
//         if (b_mul->binary_op() == binary_mul)
//         {
//             if (try_get_direct_parent<conv2d>(*b_mul, 0) || try_get_direct_parent<conv2d>(*b_mul, 1))
//                 return false;
//             if (auto b_add = try_get_direct_child<binary>(*b_mul))
//             {
//                 if (b_add->output().connections().size() == 1 && b_add->output().connections()[0]->owner().runtime_opcode() == op_conv2d)
//                 {
//                     if (b_add->binary_op() == binary_add)
//                     {
//                         if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 1))
//                         {
//                             context.inputs.emplace_back(&b_mul->input_a());
//                             context.inputs.emplace_back(&b_mul->input_b());
//                             context.matched_nodes.emplace_back(mul_const);
//                         }
//                         else if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 0))
//                         {
//                             context.inputs.emplace_back(&b_mul->input_b());
//                             context.inputs.emplace_back(&b_mul->input_a());
//                             context.matched_nodes.emplace_back(mul_const);
//                         }
//                         else
//                         {
//                             return false;
//                         }
//                         if (auto add_const = try_get_direct_parent<constant>(*b_add, 1))
//                         {
//                             context.inputs.emplace_back(&b_mul->input_b());
//                             context.matched_nodes.emplace_back(add_const);
//                         }
//                         else if (auto add_const = try_get_direct_parent<constant>(*b_add, 0))
//                         {
//                             context.inputs.emplace_back(&b_mul->input_a());
//                             context.matched_nodes.emplace_back(add_const);
//                         }
//                         else
//                         {
//                             return false;
//                         }

//                         context.outputs.emplace_back(&b_add->output());
//                         context.matched_nodes.emplace_back(b_mul);
//                         return true;
//                     }
//                 }
//             }
//         }
//     }

//     return false;
// }

// void fold_mul_add_to_dw_transform::process(transform_context &context)
// {
//     auto &output = *context.inputs[0]->connection();
//     auto inputs = context.outputs[0]->connections();
//     auto &b_mul = static_cast<binary &>(*context.matched_nodes[0]);

//     std::vector<float> ones(int(output.shape()[1]), 1.f);
//     std::vector<float> zeros(int(output.shape()[1]), 0.f);

//     auto one = context.graph.emplace<constant>(dt_float32, shape_t { output.shape()[1], 1, 1, 1 }, ones);
//     auto zero = context.graph.emplace<constant>(dt_float32, shape_t { output.shape()[1] }, zeros);
//     auto groups = (int32_t)output.shape()[1];

//     auto conv = context.graph.emplace<conv2d>(output.shape(), one->output().shape(), groups,
//         padding { 0, 0 }, padding { 0, 0 }, 1, 1, 1, 1, value_range<float>::full());

//     conv->name(b_mul.name() + "/fused_dw");
//     conv->input().connect(output);
//     conv->weights().connect(one->output());
//     conv->bias().connect(zero->output());

//     context.inputs[0]->connect(conv->output());
// }
