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
#include "../tflite_importer.h"
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SLICE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto begin = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto size = load_axis<int32_t>(get_tensor(op.inputs(), 2));
    axis_t end(begin.size());
    for (size_t i = 0; i < begin.size(); i++)
        end[i] = begin[i] + size[i];

    [[maybe_unused]] auto &options = *op.builtin_options_as_SliceOptions();
    auto node = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(STRIDED_SLICE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto begin = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto end = load_axis<int32_t>(get_tensor(op.inputs(), 2));
    auto strides = load_axis<int32_t>(get_tensor(op.inputs(), 3));
    auto &options = *op.builtin_options_as_StridedSliceOptions();
    auto op_name = std::string(get_tensor(op.outputs(), 0).name()->string_view());
#if 1
    // for (int i = 0; i < begin.size(); i++)
    // {
    //     if (begin[i] < 0 && (int32_t)(get_shape(input.shape())[i] + begin[i] + 1 > end[i]))
    //     {
    //         for (int j = 0; j < begin.size(); j++)
    //         {
    //             if (i == j)
    //             {
    //                 end[i] = (int32_t)(get_shape(input.shape())[i]) + begin[i] + 1 - end[i];
    //                 begin[i] = 0 - 1 - begin[i];
    //                 strides[i] *= -1;
    //                 // std::swap(begin[i], end[i]);
    //             }
    //             else
    //             {
    //                 end[j] = (int32_t)(get_shape(input.shape())[j]) - end[j];
    //             }
    //         }
    //     }
    // }
    // // slice *source_slice;
    // // source_slice = graph_.emplace<slice>(out_conn[out_conn.size() - 1]->type(), out_conn[out_conn.size() - 1]->shape(), begin, end, strides, options.begin_mask(),
    // //     options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
    // // source_slice->input().connect(*out_conn[out_conn.size() - 1]);

    auto source_slice = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
        options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
    source_slice->name(get_tensor(op.outputs(), 0).name()->string_view());
    auto rshape = graph_.emplace<bitcast>(source_slice->output().type(), source_slice->output().shape(), get_shape(output.shape()));
    rshape->name(source_slice->name() + "/reshape");
    rshape->input().connect(source_slice->output());

    link_input_tensor(&source_slice->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &rshape->output());
#else
#if 0
        //? Plan A: pad + slice
        std::vector<input_connector *> in_conn;
        std::vector<output_connector *> out_conn;

        std::vector<node *> node_list;
        for (int i = 0; i < begin.size(); i++)
        {
            if (begin[i] < 0 && (int32_t)(get_shape(input.shape())[i] + begin[i] + 1 > end[i]))
            {
                xt::svector<padding> paddings;
                axis_t new_begin;
                axis_t new_end;
                for (int j = 0; j < begin.size(); j++)
                {
                    if (i == j)
                    {
                        paddings.push_back(padding { 0, (int32_t)(get_shape(input.shape())[i]) });
                        new_begin.push_back((int32_t)(get_shape(input.shape())[i]) - 1 - begin[i]);
                        new_end.push_back((int32_t)(get_shape(input.shape())[i]) * 2);
                        end[i] = (int32_t)(get_shape(input.shape())[i]) + begin[i] + 1 - end[i];
                        begin[i] = 0 - 1 - begin[i];
                        // std::swap(begin[i], end[i]);
                    }
                    else
                    {
                        paddings.push_back(padding::zero());
                        new_begin.push_back(0);
                        new_end.push_back((int32_t)(get_shape(input.shape())[j]));
                    }
                }
                NNCASE_UNUSED pad *reverse_pad;
                NNCASE_UNUSED slice *reverse_slice;
                if (out_conn.empty())
                {
                    reverse_pad = graph_.emplace<pad>(to_data_type(input.type()), get_shape(input.shape()), paddings, pad_symmetric, scalar(0.f));
                }
                else
                {
                    reverse_pad = graph_.emplace<pad>(out_conn[out_conn.size() - 1]->type(), out_conn[out_conn.size() - 1]->shape(), paddings, pad_symmetric, scalar(0.f));
                }
                reverse_pad->name(op_name + "_reverse_pad");
                reverse_slice = graph_.emplace<slice>(reverse_pad->output().type(), reverse_pad->output().shape(), new_begin, new_end);
                reverse_slice->name(op_name + "_reverse_slice");
                reverse_slice->input().connect(reverse_pad->output());

                in_conn.push_back(&reverse_pad->input());
                out_conn.push_back(&reverse_slice->output());
                node_list.push_back(reverse_pad);
                node_list.push_back(reverse_slice);
            }
        }
        slice *source_slice;
        if (out_conn.empty())
        {
            source_slice = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
                options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
        }
        else
        {
            source_slice = graph_.emplace<slice>(out_conn[out_conn.size() - 1]->type(), out_conn[out_conn.size() - 1]->shape(), begin, end, strides, options.begin_mask(),
                options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
            source_slice->input().connect(*out_conn[out_conn.size() - 1]);
        }
        // auto node = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
        //     options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
        source_slice->name(get_tensor(op.outputs(), 0).name()->string_view());
        auto rshape = graph_.emplace<bitcast>(source_slice->output().type(), source_slice->output().shape(), get_shape(output.shape()));
        rshape->name(source_slice->name() + "/reshape");
        rshape->input().connect(source_slice->output());

        in_conn.push_back(&source_slice->input());
        out_conn.push_back(&rshape->output());

        link_input_tensor(in_conn[0], op.inputs()->Get(0));
        link_output_tensor(op.outputs()->Get(0), out_conn[out_conn.size() - 1]);
#else
    //? Plan B: slice
    // ! If any "-1" exist in begin or stride, reverse first.
    std::vector<input_connector *> in_conn;
    std::vector<output_connector *> out_conn;

    std::vector<node *> node_list;
    for (int i = 0; i < begin.size(); i++)
    {
        if (begin[i] < 0 && (int32_t)(get_shape(input.shape())[i] + begin[i] + 1 > end[i]))
        {

            std::vector<shape_t> slices_shape;
            std::vector<slice *> slices;
            shape_t new_slice_shape;
            axis_t slice_begin, slice_end;
            for (int j = 0; j < begin.size(); j++)
            {
                if (i == j)
                {
                    end[i] = (int32_t)(get_shape(input.shape())[i]) + begin[i] + 1 - end[i];
                    begin[i] = 0 - 1 - begin[i];
                    // std::swap(begin[i], end[i]);
                    new_slice_shape.push_back(1);
                }
                else
                {
                    new_slice_shape.push_back((int32_t)(get_shape(input.shape())[j]));
                }
                slice_begin.push_back(0);
                slice_end.push_back((int32_t)(get_shape(input.shape())[j]));
            }
            // for (int t = (int32_t)(get_shape(input.shape())[i]) - 1; t >= 0; t--)
            // {
            // }

            for (int t = (int32_t)(get_shape(input.shape())[i]) - 1; t >= 0; t--)
            {
                slices_shape.emplace_back(new_slice_shape);
                slice_begin[i] = t;
                slice_end[i] = t + 1;
                auto slc = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), slice_begin, slice_end);
                slices.push_back(slc);
                in_conn.push_back(&slc->input());
            }
            auto cct = graph_.emplace<concat>(to_data_type(input.type()), slices_shape, i);

            for (size_t s = 0; s < slices.size(); s++)
            {
                cct->input_at(s).connect(slices[s]->output());
            }
            out_conn.push_back(&cct->output());
        }
    }
    slice *source_slice;
    if (out_conn.empty())
    {
        source_slice = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
            options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
    }
    else
    {
        source_slice = graph_.emplace<slice>(out_conn[out_conn.size() - 1]->type(), out_conn[out_conn.size() - 1]->shape(), begin, end, strides, options.begin_mask(),
            options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
        source_slice->input().connect(*out_conn[out_conn.size() - 1]);
    }
    // auto node = graph_.emplace<slice>(to_data_type(input.type()), get_shape(input.shape()), begin, end, strides, options.begin_mask(),
    //     options.end_mask(), options.ellipsis_mask(), options.new_axis_mask());
    source_slice->name(get_tensor(op.outputs(), 0).name()->string_view());
    auto rshape = graph_.emplace<bitcast>(source_slice->output().type(), source_slice->output().shape(), get_shape(output.shape()));
    rshape->name(source_slice->name() + "/reshape");
    rshape->input().connect(source_slice->output());

    if (in_conn.empty())
        link_input_tensor(&source_slice->input(), op.inputs()->Get(0));
    else
    {
        for (auto &it : in_conn)
            link_input_tensor(it, op.inputs()->Get(0));
    }
    out_conn.push_back(&rshape->output());

    link_output_tensor(op.outputs()->Get(0), out_conn[out_conn.size() - 1]);
#endif
#endif
}
