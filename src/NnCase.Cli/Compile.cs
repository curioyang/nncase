﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Evaluation;
using NnCase.Evaluation.Data;
using NnCase.Importer;
using NnCase.IR;

namespace NnCase.Cli
{
    public class Compile
    {
        private readonly IServiceProvider _serviceProvider;

        public Compile(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public async Task Run(Options options)
        {
            // 1. Import
            var graph = await ImportGraph(options);

            // 2. Optimize Pass 1
            OptimizePass1(graph);

            // 3. Quantize
            await Quantize(options, graph);
        }

        private async Task<Graph> ImportGraph(Options options)
        {
            var model = await File.ReadAllBytesAsync(options.Input);
            var graph = new Graph();

            switch (options.InputFormat)
            {
                case "tflite":
                    new TFLiteImporter(_serviceProvider, model, graph).Import();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input format: {options.InputFormat}");
            }

            using (var stream = File.Create("ir.dot"))
            {
                graph.DumpDotGraph(stream);
            }

            return graph;
        }

        private void OptimizePass1(Graph graph)
        {
        }

        private async Task Quantize(Options options, Graph graph)
        {
            // 3.1. Add quantization checkpoints
            AddQuantizationCheckpoints(graph);

            // 3.2 Get activation ranges
            await GetActivationRanges(options, graph);
        }

        private void AddQuantizationCheckpoints(Graph graph)
        {
        }

        private async Task GetActivationRanges(Options options, Graph graph)
        {
            var allocators = new Dictionary<MemoryType, MemoryAllocator>
            {
                { MemoryType.Constant, new MemoryAllocator() },
                { MemoryType.Main, new MemoryAllocator() }
            };
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(graph.Outputs, allocationContext, computeSequence);

            var evaluator = new Evaluator(allocators, allocationContext.Allocations, computeSequence);

            var dataset = new ImageDataset(options.Dataset, graph.Inputs[0].Output.Shape, 0.0f, 1.0f);
            await foreach (var batch in dataset.GetBatchesAsync())
            {
                EvaluateBatch(batch, evaluator);
            }
        }

        private void EvaluateBatch(DenseTensor<float> batch, Evaluator evaluator)
        {
            var input = evaluator.InputAt<float>(0);
            batch.Buffer.Span.CopyTo(input);
        }
    }
}
