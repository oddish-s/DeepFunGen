namespace app;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

public sealed class VideoProcessingService : IDisposable
{
    private const float NormalizationMean = 0.0f;
    private const float NormalizationStd = 10.0f;

    private readonly Queue<VideoJob> _queue = new();
    private readonly SemaphoreSlim _signal = new(0);
    private readonly CancellationTokenSource _cts = new();
    private readonly SynchronizationContext? _syncContext;
    private readonly object _modelLock = new();
    private readonly object _queueLock = new();
    private readonly PostProcessor _postProcessor = new();

    private string? _modelPath;
    private string _executionProvider = "-";
    private PostprocessOptions _postprocessOptions = PostprocessOptions.Default;
    private Task? _processingTask;
    private OnnxSequenceModel? _model;

    public VideoProcessingService()
    {
        _syncContext = SynchronizationContext.Current;
    }

    public event Action<VideoJob?, string>? Log;
    public event Action<string>? ExecutionProviderChanged;
    public event Action<PredictionSummary>? PredictionReady;

    public PostprocessOptions PostprocessOptions
    {
        get => _postprocessOptions;
        set => _postprocessOptions = value ?? PostprocessOptions.Default;
    }

    public string ExecutionProvider => _executionProvider;

    public void UpdateModelPath(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path must be provided", nameof(modelPath));
        }

        var resolved = Path.GetFullPath(modelPath);
        if (!File.Exists(resolved))
        {
            throw new FileNotFoundException("ONNX model not found", resolved);
        }

        lock (_modelLock)
        {
            if (string.Equals(_modelPath, resolved, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            _model?.Dispose();
            _model = null;
            _modelPath = resolved;
        }

        NotifyExecutionProviderChanged(_executionProvider);
        PostLog(null, $"ONNX model set to {Path.GetFileName(resolved)}");
    }

    public void Enqueue(VideoJob job)
    {
        lock (_queueLock)
        {
            _queue.Enqueue(job);
        }
        _signal.Release();
        EnsureProcessingLoop();
    }

    public void ClearPending()
    {
        lock (_queueLock)
        {
            _queue.Clear();
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _signal.Release();
        try
        {
            _processingTask?.Wait();
        }
        catch (AggregateException)
        {
            // Cancellation path
        }

        _signal.Dispose();
        _cts.Dispose();
        lock (_modelLock)
        {
            _model?.Dispose();
            _model = null;
        }
    }

    private void EnsureProcessingLoop()
    {
        if (_processingTask is not null)
        {
            return;
        }

        _processingTask = Task.Run(() => ProcessQueueAsync(_cts.Token));
    }

    private async Task ProcessQueueAsync(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                await _signal.WaitAsync(token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            VideoJob? job;
            while ((job = Dequeue()) is not null)
            {
                try
                {
                    await ProcessJobAsync(job, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    job.Status = VideoJobStatus.Failed;
                    job.Message = "Cancelled";
                    PostLog(job, "Processing cancelled");
                    return;
                }
                catch (Exception ex)
                {
                    job.Status = VideoJobStatus.Failed;
                    job.Message = ex.Message;
                    PostLog(job, $"Error: {ex.Message}");
                }
            }
        }
    }

    private VideoJob? Dequeue()
    {
        lock (_queueLock)
        {
            return _queue.Count > 0 ? _queue.Dequeue() : null;
        }
    }

    private async Task ProcessJobAsync(VideoJob job, CancellationToken token)
    {
        string? modelPath;
        lock (_modelLock)
        {
            modelPath = _modelPath;
        }

        if (modelPath is null)
        {
            throw new InvalidOperationException("Select an ONNX model before processing videos.");
        }

        string predictionPath = ResolvePredictionPath(job, modelPath);
        job.PredictionPath = predictionPath;
        if (File.Exists(predictionPath) && TryLoadExistingPredictions(job, predictionPath))
        {
            return;
        }

        OnnxSequenceModel model;
        lock (_modelLock)
        {
            _model ??= new OnnxSequenceModel(modelPath, preferGpu: true);
            model = _model;
            if (!string.Equals(_executionProvider, model.ExecutionProvider, StringComparison.OrdinalIgnoreCase))
            {
                _executionProvider = model.ExecutionProvider;
                NotifyExecutionProviderChanged(_executionProvider);
                PostLog(job, $"ONNX execution provider: {_executionProvider}");
            }
        }

        job.Status = VideoJobStatus.Processing;
        job.Message = "Processing";
        UpdateProgress(job, 0.0, job.Message);
        PostLog(job, "Processing started");

        var data = await Task.Run(() => ProcessVideo(job, model, predictionPath, token), token).ConfigureAwait(false);

        if (job.Status == VideoJobStatus.Processing)
        {
            job.Status = VideoJobStatus.Completed;
            job.Message = $"Completed ({data.FrameCount} frames)";
            PostLog(job, $"Saved predictions to {Path.GetFileName(predictionPath)}");
            RunPostprocess(job, predictionPath, data);
        }
    }

    private PredictionData ProcessVideo(VideoJob job, OnnxSequenceModel model, string predictionPath, CancellationToken token)
    {
        using var capture = new VideoCapture(job.VideoPath);
        if (!capture.IsOpened())
        {
            throw new InvalidOperationException($"Failed to open video: {job.VideoPath}");
        }

        int perFrameLength = OnnxSequenceModel.Channels * OnnxSequenceModel.Height * OnnxSequenceModel.Width;
        int sequenceLength = perFrameLength * OnnxSequenceModel.SequenceLength;
        float[] sequenceBuffer = new float[sequenceLength];
        var window = new Queue<float[]>(OnnxSequenceModel.SequenceLength);
        double fps = capture.Fps;
        if (fps <= 1e-3)
        {
            fps = 30.0;
        }

        int estimatedFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);
        double totalEstimate = estimatedFrames > 0 ? estimatedFrames : double.NaN;
        var predictedChange = new List<float>(estimatedFrames > 0 ? estimatedFrames : 1024);

        using var frame = new Mat();
        int frameIndex = 0;
        double lastProgress = double.NaN;

        while (true)
        {
            token.ThrowIfCancellationRequested();
            if (!capture.Read(frame))
            {
                break;
            }

            if (frame.Empty())
            {
                continue;
            }

            if (predictedChange.Count <= frameIndex)
            {
                predictedChange.AddRange(Enumerable.Repeat(0f, frameIndex - predictedChange.Count + 1));
            }

            using var resized = new Mat();
            Cv2.Resize(frame, resized, new Size(OnnxSequenceModel.Width, OnnxSequenceModel.Height));
            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
            using var floatMat = new Mat();
            rgb.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

            float[] frameTensor = ArrayPool<float>.Shared.Rent(perFrameLength);
            bool enqueued = false;
            try
            {
                CopyToTensor(floatMat, frameTensor, perFrameLength);
                window.Enqueue(frameTensor);
                enqueued = true;
                if (window.Count > OnnxSequenceModel.SequenceLength)
                {
                    var recycled = window.Dequeue();
                    ArrayPool<float>.Shared.Return(recycled);
                }

                if (window.Count == OnnxSequenceModel.SequenceLength)
                {
                    int offset = 0;
                    foreach (var segment in window)
                    {
                        Array.Copy(segment, 0, sequenceBuffer, offset, perFrameLength);
                        offset += perFrameLength;
                    }

                    float normalized = model.Invoke(sequenceBuffer);
                    float denormalized = normalized * NormalizationStd + NormalizationMean;
                    predictedChange[frameIndex] = denormalized;
                }
            }
            finally
            {
                if (!enqueued)
                {
                    ArrayPool<float>.Shared.Return(frameTensor);
                }
            }

            frameIndex++;

            if (!double.IsNaN(totalEstimate))
            {
                double progress = Math.Clamp(frameIndex / totalEstimate, 0d, 1d);
                if (double.IsNaN(lastProgress) || progress - lastProgress >= 0.02 || frameIndex == estimatedFrames)
                {
                    UpdateProgress(job, progress, $"Processing ({frameIndex}/{estimatedFrames})");
                    lastProgress = progress;
                }
            }
            else if (frameIndex == 1 || frameIndex % 30 == 0)
            {
                UpdateProgress(job, double.NaN, $"Processing ({frameIndex} frames)");
            }
        }

        while (window.Count > 0)
        {
            var recycled = window.Dequeue();
            ArrayPool<float>.Shared.Return(recycled);
        }

        int totalFrames = frameIndex;
        if (totalFrames == 0)
        {
            throw new InvalidOperationException("No frames decoded from video");
        }

        if (predictedChange.Count < totalFrames)
        {
            predictedChange.AddRange(Enumerable.Repeat(0f, totalFrames - predictedChange.Count));
        }

        float[] changeArray = predictedChange.ToArray();
        var predictedValue = new float[totalFrames];
        var timestamps = new double[totalFrames];
        for (int i = 0; i < totalFrames; i++)
        {
            float changeValue = i < changeArray.Length ? changeArray[i] : 0f;
            if (i < OnnxSequenceModel.SequenceLength - 1)
            {
                changeValue = 0f;
                changeArray[i] = 0f;
            }
            predictedValue[i] = i == 0 ? changeValue : predictedValue[i - 1] + changeValue;
            timestamps[i] = i * (1000.0 / fps);
        }

        WriteCsv(predictionPath, changeArray, predictedValue, timestamps);
        return new PredictionData(totalFrames, timestamps, changeArray, predictedValue, fps);
    }

    private bool TryLoadExistingPredictions(VideoJob job, string predictionPath)
    {
        try
        {
            var timestamps = new List<double>();
            var changes = new List<float>();
            var values = new List<float>();
            using var reader = new StreamReader(predictionPath, Encoding.UTF8, true);
            string? line = reader.ReadLine();
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split(',');
                if (parts.Length < 4)
                {
                    continue;
                }
                if (!double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var timestamp))
                {
                    continue;
                }
                if (!float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var change))
                {
                    continue;
                }
                if (!float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    continue;
                }

                timestamps.Add(timestamp);
                changes.Add(change);
                values.Add(value);
            }

            if (timestamps.Count == 0)
            {
                return false;
            }

            double frameRate = EstimateFrameRate(timestamps);
            job.Status = VideoJobStatus.Completed;
            UpdateProgress(job, 1.0, $"Loaded predictions ({timestamps.Count} frames)");
            PostLog(job, $"Loaded predictions from {Path.GetFileName(predictionPath)}");

            var data = new PredictionData(timestamps.Count, timestamps.ToArray(), changes.ToArray(), values.ToArray(), frameRate);
            RunPostprocess(job, predictionPath, data);
            return true;
        }
        catch (Exception ex)
        {
            PostLog(job, $"Failed to read existing predictions ({ex.Message}). Recomputing...");
            return false;
        }
    }

    private void RunPostprocess(VideoJob job, string predictionPath, PredictionData data)
    {
        string directory = Path.GetDirectoryName(job.VideoPath) ?? ".";
        string videoName = Path.GetFileNameWithoutExtension(job.VideoPath);
        string scriptPath = Path.Combine(directory, $"{videoName}.funscript");

        var result = _postProcessor.Process(data.PredictedChanges, data.FrameRate, PostprocessOptions);
        var generatorModel = _modelPath is null ? "unknown" : Path.GetFileNameWithoutExtension(_modelPath);
        WriteFunscript(scriptPath, result, data.Timestamps, PostprocessOptions, generatorModel);
        UpdateProgress(job, job.Progress, $"Postprocessed ({Path.GetFileName(scriptPath)})");

        var processedValues = result.ProcessedValue.Select(d => (float)d).ToArray();
        var processedChanges = result.ProcessedChange.Select(d => (float)d).ToArray();
        var phaseMarker = result.PhaseMarker.Select(d => (float)d).ToArray();
        var summary = new PredictionSummary(
            job,
            predictionPath,
            scriptPath,
            data.Timestamps,
            data.PredictedChanges,
            data.PredictedValues,
            processedChanges,
            processedValues,
            phaseMarker,
            result.PhaseSource.ToArray(),
            PostprocessOptions,
            result);

        job.LatestSummary = summary;
        NotifyPredictionReady(summary);
    }

    private void NotifyPredictionReady(PredictionSummary summary)
    {
        void Invoke() => PredictionReady?.Invoke(summary);

        if (_syncContext is null)
        {
            Invoke();
        }
        else
        {
            _syncContext.Post(_ => Invoke(), null);
        }
    }

    private static void CopyToTensor(Mat floatMat, float[] buffer, int usableLength)
    {
        var indexer = floatMat.GetGenericIndexer<Vec3f>();
        int height = floatMat.Rows;
        int width = floatMat.Cols;
        int area = height * width;
        if (usableLength < area * OnnxSequenceModel.Channels)
        {
            throw new ArgumentException("Destination buffer is too small", nameof(buffer));
        }

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                Vec3f pixel = indexer[h, w];
                int idx = h * width + w;
                buffer[idx] = pixel.Item0;
                buffer[area + idx] = pixel.Item1;
                buffer[2 * area + idx] = pixel.Item2;
            }
        }
    }

    private void UpdateProgress(VideoJob job, double progress, string? message = null)
    {
        void Apply()
        {
            job.Progress = progress;
            if (message is not null)
            {
                job.Message = message;
            }
        }

        if (_syncContext is null)
        {
            Apply();
        }
        else
        {
            _syncContext.Post(_ => Apply(), null);
        }
    }

    private void NotifyExecutionProviderChanged(string provider)
    {
        void Invoke() => ExecutionProviderChanged?.Invoke(provider);
        if (_syncContext is null)
        {
            Invoke();
        }
        else
        {
            _syncContext.Post(_ => Invoke(), null);
        }
    }

    private void PostLog(VideoJob? job, string message)
    {
        void Invoke() => Log?.Invoke(job, message);

        if (_syncContext is null)
        {
            Invoke();
        }
        else
        {
            _syncContext.Post(_ => Invoke(), null);
        }
    }

    private static string ResolvePredictionPath(VideoJob job, string modelPath)
    {
        string directory = Path.GetDirectoryName(job.VideoPath) ?? ".";
        string videoName = Path.GetFileNameWithoutExtension(job.VideoPath);
        string modelName = Path.GetFileNameWithoutExtension(modelPath);
        string fileName = $"{videoName}.{modelName}.csv";
        return Path.Combine(directory, fileName);
    }

    private static double EstimateFrameRate(IReadOnlyList<double> timestamps)
    {
        if (timestamps.Count < 2)
        {
            return 30.0;
        }
        double delta = timestamps[1] - timestamps[0];
        if (delta <= 0)
        {
            return 30.0;
        }
        return 1000.0 / delta;
    }

    private static void WriteCsv(string outputPath, IReadOnlyList<float> predictedChange, IReadOnlyList<float> predictedValue, IReadOnlyList<double> timestamps)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("frame_index,timestamp_ms,predicted_change,predicted_value");
        var culture = CultureInfo.InvariantCulture;
        int frameCount = predictedValue.Count;
        for (int i = 0; i < frameCount; i++)
        {
            double timestamp = i < timestamps.Count ? timestamps[i] : 0.0;
            float change = i < predictedChange.Count ? predictedChange[i] : 0f;
            float value = predictedValue[i];
            string line = string.Format(
                culture,
                "{0},{1:G17},{2:G9},{3:G9}",
                i,
                timestamp,
                change,
                value);
            writer.WriteLine(line);
        }
    }

    public sealed record PredictionSummary(
        VideoJob Job,
        string PredictionPath,
        string ScriptPath,
        double[] Timestamps,
        float[] PredictedChanges,
        float[] PredictedValues,
        float[] ProcessedChanges,
        float[] ProcessedValues,
        float[] PhaseMarker,
        string[] PhaseSource,
        PostprocessOptions Options,
        PostprocessResult DetailedResult);

    private static void WriteFunscript(string path, PostprocessResult result, IReadOnlyList<double> timestamps, PostprocessOptions options, string modelName)
    {
        var file = new FunscriptFile
        {
            Actions = BuildActions(result.GraphPoints, timestamps),
            Inverted = false,
            Range = 100,
            Version = "1.0",
            Generator = new FunscriptGenerator
            {
                Name = "DeepFunGen",
                Version = "0.1.0",
                Model = string.IsNullOrWhiteSpace(modelName) ? "unknown" : modelName,
                Options = CloneOptions(options)
            }
        };

        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var generatorOptions = new JsonSerializerOptions
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        var compactArrayOptions = new JsonSerializerOptions
        {
            WriteIndented = false,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        using var stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        var actionsJson = JsonSerializer.SerializeToUtf8Bytes(file.Actions, compactArrayOptions);

        writer.WriteStartObject();
        writer.WritePropertyName("actions");
        writer.WriteRawValue(actionsJson);
        writer.WriteBoolean("inverted", file.Inverted);
        writer.WriteNumber("range", file.Range);
        writer.WriteString("version", file.Version);
        writer.WritePropertyName("generator");
        JsonSerializer.Serialize(writer, file.Generator, generatorOptions);
        writer.WriteEndObject();
    }

    private static List<FunscriptAction> BuildActions(IReadOnlyList<GraphPointInfo> graphPoints, IReadOnlyList<double> timestamps)
    {
        var actions = new List<FunscriptAction>();
        if (graphPoints is not { Count: > 0 })
        {
            int fallbackTime = timestamps.Count > 0 ? (int)Math.Round(timestamps[0]) : 0;
            actions.Add(new FunscriptAction(Math.Max(0, fallbackTime), 50));
            return actions;
        }

        double defaultDelta = timestamps.Count > 1 ? Math.Max(1.0, Math.Abs(timestamps[1] - timestamps[0])) : 1000.0 / 30.0;
        int lastTimeMs = -1;

        foreach (var point in graphPoints.OrderBy(p => p.Position))
        {
            if (double.IsNaN(point.Value) || double.IsInfinity(point.Value))
            {
                continue;
            }

            int frameIndex = (int)Math.Round(point.Position);
            double timestamp = EstimateTimestamp(frameIndex, timestamps, defaultDelta);
            int timeMs = (int)Math.Round(timestamp);
            if (timeMs < 0)
            {
                timeMs = 0;
            }
            if (lastTimeMs >= 0 && timeMs <= lastTimeMs)
            {
                timeMs = lastTimeMs + 1;
            }

            int amplitude = (int)Math.Round(point.Value);
            amplitude = Math.Clamp(amplitude, 0, 100);

            actions.Add(new FunscriptAction(timeMs, amplitude));
            lastTimeMs = timeMs;
        }

        if (actions.Count == 0)
        {
            int fallbackTime = timestamps.Count > 0 ? (int)Math.Round(timestamps[0]) : 0;
            actions.Add(new FunscriptAction(Math.Max(0, fallbackTime), 50));
        }

        return actions;
    }

    private static double EstimateTimestamp(int frameIndex, IReadOnlyList<double> timestamps, double defaultDelta)
    {
        if (timestamps.Count == 0)
        {
            return Math.Max(0, frameIndex) * defaultDelta;
        }

        if (frameIndex < 0)
        {
            return Math.Max(0.0, timestamps[0] + frameIndex * defaultDelta);
        }

        if (frameIndex < timestamps.Count)
        {
            return timestamps[frameIndex];
        }

        int lastIndex = timestamps.Count - 1;
        double lastTimestamp = timestamps[lastIndex];
        return lastTimestamp + (frameIndex - lastIndex) * defaultDelta;
    }

    private static PostprocessOptions CloneOptions(PostprocessOptions source)
    {
        source ??= PostprocessOptions.Default;
        return new PostprocessOptions
        {
            SmoothWindowFrames = source.SmoothWindowFrames,
            ProminenceRatio = source.ProminenceRatio,
            MinProminence = source.MinProminence,
            MaxSlope = source.MaxSlope,
            BoostSlope = source.BoostSlope,
            MinSlope = source.MinSlope,
            MergeThresholdMs = source.MergeThresholdMs,
            CentralDeviationThreshold = source.CentralDeviationThreshold
        };
    }

    private sealed class FunscriptFile
    {
        [JsonPropertyName("actions")]
        public List<FunscriptAction> Actions { get; set; } = new();

        [JsonPropertyName("inverted")]
        public bool Inverted { get; set; }

        [JsonPropertyName("range")]
        public int Range { get; set; }

        [JsonPropertyName("version")]
        public string Version { get; set; } = "1.0";


        [JsonPropertyName("generator")]
        public FunscriptGenerator Generator { get; set; } = new();
    }

    private sealed class FunscriptGenerator
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = "DeepFunGen";

        [JsonPropertyName("version")]
        public string Version { get; set; } = "0.1.0";

        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("options")]
        public PostprocessOptions Options { get; set; } = PostprocessOptions.Default;
    }

    private sealed record FunscriptAction(
        [property: JsonPropertyName("at")] int At,
        [property: JsonPropertyName("pos")] int Pos);

    private readonly record struct PredictionData(
        int FrameCount,
        double[] Timestamps,
        float[] PredictedChanges,
        float[] PredictedValues,
        double FrameRate);
}















