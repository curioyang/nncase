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
            if (auto b_add = try_get_direct_child<binary>(*b_mul))
            {
                if (b_add->output().connections().size() == 1 && b_add->output().connections()[0]->owner().runtime_opcode() == op_conv2d)
                {
                    if (b_add->binary_op() == binary_add)
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
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &b_mul = static_cast<binary &>(*context.matched_nodes[0]);
    auto &b_add = static_cast<binary &>(*context.matched_nodes[1]);

    auto bitc_weights = context.graph.emplace<bitcast>(weights.type(), weights.shape(), shape_t { weights.shape()[0], 1, 1, 1 });
    auto bitc_bias = context.graph.emplace<bitcast>(bias.type(), bias.shape(), shape_t { bias.shape()[0] });

    // auto in_h = (int32_t)output.shape()[2];
    // auto in_w = (int32_t)output.shape()[3];
    auto groups = (int32_t)bitc_weights->output().shape()[0];
    // auto f_h = (int32_t)bitc_weights->output().shape()[2];
    // auto f_w = (int32_t)bitc_weights->output().shape()[3];
    // auto stride_h = 1;
    // auto stride_w = 1;
    // auto dilation_h = 1;
    // auto dilation_w = 1;
    // auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, 0);
    // auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, 0);
    // conv2d *conv;
    auto clamp = b_add.fused_activation();

    auto conv = context.graph.emplace<conv2d>(output.shape(), bitc_weights->output().shape(), groups,
        padding { 0, 0 }, padding { 0, 0 }, 1, 1, 1, 1, clamp);
    // conv->name(get_tensor(op.outputs(), 0).name()->string_view());

    // auto conv = context.graph.emplace<conv2d>(output.shape(), bitc_weights->output().shape(), 1, padding { 0, 0 }, padding { 0, 0 }, 1, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, clamp);

    bitc_weights->name(b_mul.name() + "/fused_dw" + "_bitc_weights");
    bitc_weights->input().connect(weights);
    bitc_bias->name(b_mul.name() + "/fused_dw" + "_bitc_bias");
    bitc_bias->input().connect(bias);
    conv->name(b_mul.name() + "/fused_dw");
    conv->input().connect(output);
    conv->weights().connect(bitc_weights->output());
    conv->bias().connect(bitc_bias->output());

    for (auto &in : dup(inputs))
        in->connect(conv->output());
}
