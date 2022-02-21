// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Transform;

/// <summary>
/// Transform module.
/// </summary>
public class TransformModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<RewriteProvider>().AsImplementedInterfaces().SingleInstance();
    }
}
