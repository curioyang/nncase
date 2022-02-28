﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Imaging;
using Nncase.IR.NN;
using Nncase.IR.Random;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

/// <summary>
/// Imaging functional helper.
/// </summary>
public static class Imaging
{
    /// <summary>
    /// resize image.
    /// </summary>
    /// <param name="resizeMode"></param>
    /// <param name="input"></param>
    /// <param name="newSize"></param>
    /// <param name="alignCorners"></param>
    /// <param name="halfPixelCenters"></param>
    /// <returns></returns>
    public static Call ResizeImage(ImageResizeMode resizeMode, Expr input, Expr newSize, Expr alignCorners, Expr halfPixelCenters) => new Call(new ResizeImage(resizeMode), input, newSize, alignCorners, halfPixelCenters);
}
