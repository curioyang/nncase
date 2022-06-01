/* This file is generated by tools/stackvm_gen/IsaGen at 06/01/2022 17:14:34 +08:00.
 *
 * Copyright 2019-2021 Canaan Inc.
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
#include "../runtime_function.h"
#include <nncase/kernels/stackvm/tensor_ops.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_batch_normalization_op_t &op) noexcept {
    try_var(momentum, pop_tensor());
    try_var(epsilon, pop_tensor());
    try_var(input_var, pop_tensor());
    try_var(input_mean, pop_tensor());
    try_var(bias, pop_tensor());
    try_var(scale, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::batch_normalization(input, scale, bias, input_mean, input_var, epsilon, momentum, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_batch_to_space_op_t &op) noexcept {
    try_var(crops, pop_tensor());
    try_var(block_shape, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::batch_to_space(input, block_shape, crops, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_binary_op_t &op) noexcept {
    try_var(rhs, pop_tensor());
    try_var(lhs, pop_tensor());
    try_var(output, kernels::stackvm::binary(op.binary_op, lhs, rhs, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_broadcast_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::broadcast(input, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_cast_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::cast(op.new_type, input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_celu_op_t &op) noexcept {
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::celu(input, alpha, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_clamp_op_t &op) noexcept {
    try_var(max, pop_tensor());
    try_var(min, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::clamp(input, min, max, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_compare_op_t &op) noexcept {
    try_var(rhs, pop_tensor());
    try_var(lhs, pop_tensor());
    try_var(output, kernels::stackvm::compare(op.compare_op, lhs, rhs, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_concat_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::concat(input, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_conv2d_op_t &op) noexcept {
    try_var(fused_clamp, pop_tensor());
    try_var(groups, pop_tensor());
    try_var(dilation, pop_tensor());
    try_var(padding, pop_tensor());
    try_var(stride, pop_tensor());
    try_var(bias, pop_tensor());
    try_var(weights, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::conv2d(op.pad_mode, input, weights, bias, stride, padding, dilation, groups, fused_clamp, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_conv2d_transpose_op_t &op) noexcept {
    try_var(fused_clamp, pop_tensor());
    try_var(groups, pop_tensor());
    try_var(dilation, pop_tensor());
    try_var(output_padding, pop_tensor());
    try_var(padding, pop_tensor());
    try_var(stride, pop_tensor());
    try_var(output_shape, pop_tensor());
    try_var(bias, pop_tensor());
    try_var(weights, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::conv2d_transpose(op.pad_mode, input, weights, bias, output_shape, stride, padding, output_padding, dilation, groups, fused_clamp, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_cum_sum_op_t &op) noexcept {
    try_var(reverse, pop_tensor());
    try_var(exclusive, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::cum_sum(input, axis, exclusive, reverse, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_dequantize_op_t &op) noexcept {
    try_var(dequant_param, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::dequantize(op.target_type, input, dequant_param, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_elu_op_t &op) noexcept {
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::elu(input, alpha, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_expand_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::expand(input, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_flatten_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::flatten(input, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_gather_op_t &op) noexcept {
    try_var(index, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::gather(input, axis, index, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_gather_nd_op_t &op) noexcept {
    try_var(index, pop_tensor());
    try_var(batch_dims, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::gather_nd(input, batch_dims, index, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_get_item_op_t &op) noexcept {
    try_var(index, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::get_item(input, index, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_hard_sigmoid_op_t &op) noexcept {
    try_var(beta, pop_tensor());
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::hard_sigmoid(input, alpha, beta, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_hard_swish_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::hard_swish(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_hardmax_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::hardmax(input, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_instance_normalization_op_t &op) noexcept {
    try_var(epsilon, pop_tensor());
    try_var(bias, pop_tensor());
    try_var(scale, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::instance_normalization(input, scale, bias, epsilon, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_l2_normalization_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::l2_normalization(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_leaky_relu_op_t &op) noexcept {
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::leaky_relu(input, alpha, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_log_softmax_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::log_softmax(input, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_lp_normalization_op_t &op) noexcept {
    try_var(p, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::lp_normalization(input, axis, p, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_lrn_op_t &op) noexcept {
    try_var(size, pop_tensor());
    try_var(bias, pop_tensor());
    try_var(beta, pop_tensor());
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::lrn(input, alpha, beta, bias, size, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_lstm_op_t &op) noexcept {
    try_var(output_size, pop_tensor());
    try_var(input_forget, pop_tensor());
    try_var(hidden_size, pop_tensor());
    try_var(clip, pop_tensor());
    try_var(activation_beta, pop_tensor());
    try_var(activation_alpha, pop_tensor());
    try_var(p, pop_tensor());
    try_var(initial_c, pop_tensor());
    try_var(initial_h, pop_tensor());
    try_var(sequence_lens, pop_tensor());
    try_var(b, pop_tensor());
    try_var(r, pop_tensor());
    try_var(w, pop_tensor());
    try_var(x, pop_tensor());
    try_var(output, kernels::stackvm::lstm(op.direction, op.layout, op.activations, x, w, r, b, sequence_lens, initial_h, initial_c, p, activation_alpha, activation_beta, clip, hidden_size, input_forget, output_size, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_mat_mul_op_t &op) noexcept {
    try_var(rhs, pop_tensor());
    try_var(lhs, pop_tensor());
    try_var(output, kernels::stackvm::mat_mul(lhs, rhs, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_normal_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(seed, pop_tensor());
    try_var(scale, pop_tensor());
    try_var(mean, pop_tensor());
    try_var(output, kernels::stackvm::normal(op.type, mean, scale, seed, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_normal_like_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(seed, pop_tensor());
    try_var(scale, pop_tensor());
    try_var(mean, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::normal_like(op.type, input, mean, scale, seed, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_one_hot_op_t &op) noexcept {
    try_var(off_value, pop_tensor());
    try_var(on_value, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(values, pop_tensor());
    try_var(depth, pop_tensor());
    try_var(indices, pop_tensor());
    try_var(output, kernels::stackvm::one_hot(op.one_hot_mode, indices, depth, values, axis, on_value, off_value, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_pad_op_t &op) noexcept {
    try_var(value, pop_tensor());
    try_var(pads, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::pad(op.pad_mode, input, pads, value, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_prelu_op_t &op) noexcept {
    try_var(slope, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::prelu(input, slope, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_prod_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::prod(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_quant_param_of_op_t &op) noexcept {
    try_var(bits, pop_tensor());
    try_var(range, pop_tensor());
    try_var(output, kernels::stackvm::quant_param_of(op.quant_mode, range, bits, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_quantize_op_t &op) noexcept {
    try_var(quant_param, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::quantize(op.target_type, input, quant_param, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_range_op_t &op) noexcept {
    try_var(step, pop_tensor());
    try_var(end, pop_tensor());
    try_var(begin, pop_tensor());
    try_var(output, kernels::stackvm::range(begin, end, step, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_range_of_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::range_of(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_reduce_op_t &op) noexcept {
    try_var(keep_dims, pop_tensor());
    try_var(init_value, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::reduce(op.reduce_op, input, axis, init_value, keep_dims, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_reduce_arg_op_t &op) noexcept {
    try_var(select_last_index, pop_tensor());
    try_var(keep_dims, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::reduce_arg(op.reduce_arg_op, input, axis, keep_dims, select_last_index, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_reduce_window2d_op_t &op) noexcept {
    try_var(count_include_pad, pop_tensor());
    try_var(ceil_mode, pop_tensor());
    try_var(dilation, pop_tensor());
    try_var(padding, pop_tensor());
    try_var(stride, pop_tensor());
    try_var(filter, pop_tensor());
    try_var(init_value, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::reduce_window2d(op.reduce_op, input, init_value, filter, stride, padding, dilation, ceil_mode, count_include_pad, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_relu_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::relu(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_relu6_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::relu6(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_require_op_t &op) noexcept {
    try_var(value, pop_tensor());
    try_var(predicate, pop_tensor());
    try_var(output, kernels::stackvm::require(op.message, predicate, value, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_reshape_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::reshape(input, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_resize_image_op_t &op) noexcept {
    try_var(extrapolation_value, pop_tensor());
    try_var(exclude_outside, pop_tensor());
    try_var(cubic_coeff_a, pop_tensor());
    try_var(new_size, pop_tensor());
    try_var(roi, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::resize_image(op.resize_mode, op.transformation_mode, op.nearest_mode, op.is_tfresize, input, roi, new_size, cubic_coeff_a, exclude_outside, extrapolation_value, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_reverse_sequence_op_t &op) noexcept {
    try_var(time_axis, pop_tensor());
    try_var(batch_axis, pop_tensor());
    try_var(seq_lens, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::reverse_sequence(input, seq_lens, batch_axis, time_axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_select_op_t &op) noexcept {
    try_var(false_value, pop_tensor());
    try_var(true_value, pop_tensor());
    try_var(predicate, pop_tensor());
    try_var(output, kernels::stackvm::select(predicate, true_value, false_value, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_selu_op_t &op) noexcept {
    try_var(gamma, pop_tensor());
    try_var(alpha, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::selu(input, alpha, gamma, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_shape_of_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::shape_of(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_sigmoid_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::sigmoid(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_size_of_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::size_of(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_slice_op_t &op) noexcept {
    try_var(strides, pop_tensor());
    try_var(axes, pop_tensor());
    try_var(ends, pop_tensor());
    try_var(begins, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::slice(input, begins, ends, axes, strides, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_softmax_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::softmax(input, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_softplus_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::softplus(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_softsign_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::softsign(input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_space_to_batch_op_t &op) noexcept {
    try_var(paddings, pop_tensor());
    try_var(block_shape, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::space_to_batch(input, block_shape, paddings, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_split_op_t &op) noexcept {
    try_var(sections, pop_tensor());
    try_var(axis, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::split(input, axis, sections, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_squeeze_op_t &op) noexcept {
    try_var(dim, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::squeeze(input, dim, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_stack_op_t &op) noexcept {
    try_var(axis, pop_tensor());
    try_var(inputs, pop_tensor());
    try_var(output, kernels::stackvm::stack(inputs, axis, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_tile_op_t &op) noexcept {
    try_var(repeats, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::tile(input, repeats, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_transpose_op_t &op) noexcept {
    try_var(perm, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::transpose(input, perm, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_unary_op_t &op) noexcept {
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::unary(op.unary_op, input, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_uniform_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(seed, pop_tensor());
    try_var(low, pop_tensor());
    try_var(high, pop_tensor());
    try_var(output, kernels::stackvm::uniform(op.type, high, low, seed, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_uniform_like_op_t &op) noexcept {
    try_var(shape, pop_tensor());
    try_var(seed, pop_tensor());
    try_var(low, pop_tensor());
    try_var(high, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::uniform_like(op.type, input, high, low, seed, shape, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_unsqueeze_op_t &op) noexcept {
    try_var(dim, pop_tensor());
    try_var(input, pop_tensor());
    try_var(output, kernels::stackvm::unsqueeze(input, dim, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}

result<void> stackvm_runtime_function::visit([[maybe_unused]] const tensor_where_op_t &op) noexcept {
    try_var(y, pop_tensor());
    try_var(x, pop_tensor());
    try_var(cond, pop_tensor());
    try_var(output, kernels::stackvm::where(cond, x, y, nullptr, module().kernel_context()));
    return stack_.push(std::move(output));
}
