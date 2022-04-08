// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Random = Nncase.IR.F.Random;
namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fuse <see cref="IR.NN.Pad"/> into <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FusePadConv2d : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        PadMode.Constant,
        IsPad(
            PadMode.Constant,
            IsWildcard("input"),
            IsWildcard("pads1"),
            // IsTensorConst("pads1", x =>
            // {
            //     // Get *_b index
            //     for (var i = 0; i < x.Value.Shape[0].FixedValue; i++)
            //     {
            //         // Paddings [ {n_b, n_a}, {c_b, c_a}, {h_b, h_a}, {w_b, w_a} ]
            //         if (i >= 2)
            //         {
            //             if (x.Value[i, 0] is 0 && x.Value[i, 1] is 0)
            //             {
            //                 return false;
            //             }
            //         }
            //     }
            //
            //     return true;
            // }), 
            IsWildcard("value")),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsWildcard("stride"),
        IsWildcard("pads2"),
        IsWildcard("dilation"),
        IsWildcard("groups"));

    private Expr? GetReplace(Expr input, Expr pads1, Expr weights, Expr bias, Expr stride, Expr pads2, Expr dilation,
        Expr groups)
    {
        var newPadsH = new[] {0, 0};
        var newPadsW = new[] {0, 0};

        // var NeedPaddingShape = IsConst(pads1[2]);
        // if (pads1 is not TensorConst)
        // {
            var NeedPaddingShape = pads1.Evaluate().AsTensor();
            if (NeedPaddingShape[2, 0] is 0
                && NeedPaddingShape[2, 1] is 0
                && NeedPaddingShape[3, 0] is 0
                && NeedPaddingShape[3, 1] is 0)
            {
                return null;
                // return Conv2D(Pad(input, pads1, PadMode.Constant, 0f), weights, bias, stride, pads2,
                //     dilation, PadMode.Constant, groups);
            }
        // }

        var convPadsH = Stack(new IR.Tuple(pads1[2, 0] + pads2[0, 0], pads1[2, 1] + pads2[0, 1]), 0);
        var convPadsW = Stack(new IR.Tuple(pads1[3, 0] + pads2[1, 0], pads1[3, 1] + pads2[1, 1]), 0);
        var newPads = Stack(new IR.Tuple(pads1[0], pads1[1], new int[] {0, 0}, new int[] {0, 0}), 0);
        var convPads = Stack(new IR.Tuple(convPadsH, convPadsW), 0);

        // var paddingsShape = newPads.Evaluate().AsTensor();
        // var inputShape = input;
        // var xx = ShapeOf(input);
        // if (paddingsShape[2, 0] == inputShape[2, 0]
        //     && paddingsShape[2, 1] == inputShape[2, 1]
        //     && paddingsShape[3, 0] == inputShape[3, 0]
        //     && paddingsShape[3, 1] == inputShape[3, 1])
        // {
        //     return Conv2D(input, weights, bias, stride, convPads, dilation, PadMode.Constant, groups);
        // }
        // else
        // {
        return Conv2D(Pad(input, newPads, PadMode.Constant, 0f), weights, bias, stride, convPads, dilation,
            PadMode.Constant, groups);
        // }

    }
}
