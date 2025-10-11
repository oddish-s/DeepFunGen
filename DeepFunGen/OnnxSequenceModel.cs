namespace app;

using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public sealed class OnnxSequenceModel : IDisposable
{
    public const int SequenceLength = 10;
    public const int Channels = 3;
    public const int Height = 224;
    public const int Width = 224;

    private readonly SessionOptions _options;
    private readonly InferenceSession _session;
    private readonly object _sessionLock = new();

    public string ExecutionProvider { get; }

    public OnnxSequenceModel(string modelPath, bool preferGpu = true)
    {
        if (!System.IO.File.Exists(modelPath))
        {
            throw new System.IO.FileNotFoundException("ONNX model not found", modelPath);
        }

        (_options, ExecutionProvider) = CreateSessionOptions(preferGpu);
        _session = new InferenceSession(modelPath, _options);
    }

    private static SessionOptions CreateOptimizedOptions()
    {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        return options;
    }

    private static bool TryCreateProviderOptions(Action<SessionOptions> attachProvider, string providerName, out (SessionOptions Options, string Provider) result)
    {
        SessionOptions? options = null;
        try
        {
            options = CreateOptimizedOptions();
            attachProvider(options);
            result = (options, providerName);
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or OnnxRuntimeException or EntryPointNotFoundException or InvalidOperationException)
        {
            options?.Dispose();
        }

        result = default;
        return false;
    }

    private static (SessionOptions Options, string Provider) CreateSessionOptions(bool preferGpu)
    {
        if (preferGpu)
        {
            if (TryCreateProviderOptions(static options => options.AppendExecutionProvider_DML(), "DirectML", out var directMl))
            {
                return directMl;
            }

            if (TryCreateProviderOptions(static options => options.AppendExecutionProvider_CUDA(), "CUDA", out var cuda))
            {
                return cuda;
            }
        }

        return (CreateOptimizedOptions(), "CPU");
    }

    public float Invoke(ReadOnlySpan<float> input)
    {
        int expected = SequenceLength * Channels * Height * Width;
        if (input.Length != expected)
        {
            throw new ArgumentException($"Expected flattened tensor of length {expected}", nameof(input));
        }

        var tensor = new DenseTensor<float>(new[] { 1, SequenceLength, Channels, Height, Width });
        input.CopyTo(tensor.Buffer.Span);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", tensor) };

        lock (_sessionLock)
        {
            using var results = _session.Run(inputs);
            using var first = results.First();
            var outputTensor = first.AsTensor<float>();
            return outputTensor[0];
        }
    }

    public void Dispose()
    {
        _session.Dispose();
        _options.Dispose();
    }
}
