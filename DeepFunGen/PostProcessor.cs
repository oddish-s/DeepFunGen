namespace app;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

public sealed class PostProcessor
{
    private const double PeakValue = 100.0;
    private const double TroughValue = 0.0;

    public PostprocessResult Process(ReadOnlySpan<float> change, double frameRate, PostprocessOptions options)
    {
        int n = change.Length;
        if (n == 0)
        {
            return PostprocessResult.Fallback(0, options);
        }

        frameRate = frameRate > 0 ? frameRate : 30.0;
        double[] rawValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            rawValues[i] = change[i];
        }

        double[] processed = (double[])rawValues.Clone();
        double[] denoised = Array.Empty<double>();

        if (options.FftDenoise)
        {
            processed = ApplyFftDenoise(processed, options.FftFramesPerComponent, options.FftWindowFrames);
            denoised = (double[])processed.Clone();
        }

        double[] delayProfile = Array.Empty<double>();
        DelayMarkerInfo[] delayMarkers = Array.Empty<DelayMarkerInfo>();
        double averageDelayFrames = 0.0;

        int delayFrames = (int)Math.Round(averageDelayFrames);

        int smoothWindow = Math.Max(1, options.SmoothWindowFrames);
        double[] smoothed = Smooth(processed, smoothWindow);

        var rawExtrema = DetectExtrema(smoothed, rawValues, processed, options);
        if (rawExtrema.Count == 0)
        {
            return PostprocessResult.Fallback(n, options);
        }

        var stageTwo = InsertMissingExtrema(rawExtrema, n, rawValues, processed);
        int mergeFrames = MsToFrames(options.MergeThresholdMs, frameRate);
        if (mergeFrames > 0)
        {
            stageTwo = MergeTriplets(stageTwo, mergeFrames, rawValues, processed);
        }
        stageTwo = SortExtrema(stageTwo);
        if (stageTwo.Count < 2)
        {
            return PostprocessResult.Fallback(n, options);
        }
        var graphPoints = ApplySlopeConstraints(stageTwo, options.MinSlope, options.MaxSlope, options.BoostSlope);
        if (graphPoints.Count < 2)
        {
            return PostprocessResult.Fallback(n, options);
        }

        graphPoints = BoostFlatPeaks(graphPoints, options.BoostSlope);
        graphPoints = ApplyCentralDeviationConstraint(graphPoints, stageTwo, options.CentralDeviationThreshold);
        graphPoints = DedupeGraphPoints(graphPoints);
        if (graphPoints.Count < 2)
        {
            return PostprocessResult.Fallback(n, options);
        }


        var processedValue = InterpolateGraph(graphPoints, n);
        var processedChange = new double[n];
        for (int i = 1; i < n; i++)
        {
            processedChange[i] = processedValue[i] - processedValue[i - 1];
        }

        var phaseMarker = Enumerable.Repeat(double.NaN, n).ToArray();
        var phaseSource = Enumerable.Repeat(string.Empty, n).ToArray();
        foreach (var pt in graphPoints)
        {
            if (pt.Label is "peak" or "trough")
            {
                int idx = (int)Math.Round(pt.Position);
                if ((uint)idx < (uint)n)
                {
                    phaseMarker[idx] = pt.Label == "peak" ? PeakValue : TroughValue;
                    phaseSource[idx] = pt.Origin;
                }
            }
        }

        var rawMarker = Enumerable.Repeat(double.NaN, n).ToArray();
        foreach (var ext in rawExtrema)
        {
            int idx = (int)Math.Round(ext.Frame);
            if ((uint)idx < (uint)n)
            {
                rawMarker[idx] = ext.Kind == "peak" ? PeakValue : TroughValue;
            }
        }

        var rawInfos = rawExtrema.Select(e => new ExtremumInfo(e.Frame, e.Kind, e.Origin, e.RawChange, e.Prominence)).ToArray();
        var stageInfos = stageTwo.Select(e => new ExtremumInfo(e.Frame, e.Kind, e.Origin, e.RawChange, e.Prominence)).ToArray();
        var graphInfos = graphPoints.Select(g => new GraphPointInfo(g.Position, g.Value, g.Label, g.Origin, g.Direction)).ToArray();

        return new PostprocessResult(
            rawValues,
            smoothed,
            denoised,
            rawInfos,
            stageInfos,
            graphInfos,
            delayMarkers,
            delayProfile,
            processedValue,
            processedChange,
            phaseMarker,
            phaseSource,
            rawMarker,
            delayFrames,
            options
        );
    }

    private static double[] Smooth(double[] values, int window)
    {
        int n = values.Length;
        if (window <= 1 || n == 0)
        {
            return (double[])values.Clone();
        }

        int half = window / 2;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - half);
            int end = Math.Min(n - 1, i + half);
            double sum = 0;
            int count = 0;
            for (int j = start; j <= end; j++)
            {
                sum += values[j];
                count++;
            }
            result[i] = count > 0 ? sum / count : values[i];
        }
        return result;
    }

    private static List<Extremum> DetectExtrema(double[] smoothed, double[] raw, double[] processed, PostprocessOptions options)
    {
        var results = new List<Extremum>();
        int windowSize = 300;
        int context = Math.Max(windowSize, 1);
        int step = Math.Max(1, windowSize / 2);
        int length = smoothed.Length;

        for (int start = 0; start < length; start += step)
        {
            int end = Math.Min(start + windowSize, length);
            if (end - start <= 0)
            {
                continue;
            }

            var window = new double[end - start];
            Array.Copy(smoothed, start, window, 0, window.Length);

            double windowMin = window.Min();
            double windowMax = window.Max();
            double dataRange = windowMax - windowMin;
            double threshold = Math.Max(dataRange * options.ProminenceRatio, options.MinProminence);

            int extendedStart = Math.Max(0, start - context);
            int extendedEnd = Math.Min(length, end + context);
            int extendedLength = extendedEnd - extendedStart;
            if (extendedLength <= 0)
            {
                continue;
            }

            var extended = new double[extendedLength];
            Array.Copy(smoothed, extendedStart, extended, 0, extendedLength);

            var extendedInverted = new double[extendedLength];
            for (int i = 0; i < extendedLength; i++)
            {
                extendedInverted[i] = -extended[i];
            }

            (int[] peaks, PeakProperties[] peakProps) peakResult;
            (int[] peaks, PeakProperties[] peakProps) troughResult;
            if (threshold > 0)
            {
                peakResult = FindPeaks.Find(extended, prominence: (threshold, null), width: (1.0, null));
                troughResult = FindPeaks.Find(extendedInverted, prominence: (threshold, null), width: (1.0, null));
            }
            else
            {
                peakResult = FindPeaks.Find(extended);
                troughResult = FindPeaks.Find(extendedInverted);
            }

            for (int i = 0; i < peakResult.peaks.Length; i++)
            {
                int global = extendedStart + peakResult.peaks[i];
                if (global < start || global >= end || (uint)global >= (uint)raw.Length)
                {
                    continue;
                }

                double width = Math.Max(0.0, peakResult.peakProps[i].Width);

                results.Add(new Extremum(
                    global,
                    "peak",
                    raw[global],
                    processed[(uint)global < (uint)processed.Length ? global : processed.Length - 1],
                    peakResult.peakProps[i].Prominence,
                    windowMin,
                    windowMax,
                    "original",
                    width
                ));
            }

            for (int i = 0; i < troughResult.peaks.Length; i++)
            {
                int global = extendedStart + troughResult.peaks[i];
                if (global < start || global >= end || (uint)global >= (uint)raw.Length)
                {
                    continue;
                }

                double width = Math.Max(0.0, troughResult.peakProps[i].Width);

                results.Add(new Extremum(
                    global,
                    "trough",
                    raw[global],
                    processed[(uint)global < (uint)processed.Length ? global : processed.Length - 1],
                    troughResult.peakProps[i].Prominence,
                    windowMin,
                    windowMax,
                    "original",
                    width
                ));
            }
        }

        var frameDict = new Dictionary<int, Extremum>();
        foreach (var ext in results)
        {
            int frame = (int)Math.Round(ext.Frame);
            if (!frameDict.TryGetValue(frame, out var existing) || ext.Prominence > existing.Prominence)
            {
                frameDict[frame] = ext;
            }
        }
        return SortExtrema(frameDict.Values);
    }


    private static List<Extremum> InsertMissingExtrema(IReadOnlyList<Extremum> extrema, int totalFrames, double[] rawChange, double[] processedChange)
    {
        if (extrema.Count == 0)
        {
            return new List<Extremum>();
        }
        var result = new List<Extremum>();
        for (int i = 0; i < extrema.Count; i++)
        {
            var current = extrema[i];
            result.Add(current);
            if (i == extrema.Count - 1)
            {
                continue;
            }
            var next = extrema[i + 1];
            if (current.Kind == next.Kind)
            {
                int currentFrame = (int)Math.Round(current.Frame);
                int nextFrame = (int)Math.Round(next.Frame);
                int mid = currentFrame == nextFrame ? currentFrame : (currentFrame + nextFrame) / 2;
                mid = Math.Clamp(mid, 0, totalFrames - 1);
                string opposite = current.Kind == "trough" ? "peak" : "trough";
                double estimatedWidth = 0.0;
                if (current.Width > 0.0 && next.Width > 0.0)
                {
                    estimatedWidth = (current.Width + next.Width) * 0.5;
                }
                else
                {
                    estimatedWidth = Math.Max(current.Width, next.Width);
                }
                if (estimatedWidth <= 0.0)
                {
                    estimatedWidth = 1.0;
                }
                result.Add(new Extremum(
                    mid,
                    opposite,
                    rawChange[mid],
                    processedChange[mid],
                    0.0,
                    rawChange[mid],
                    rawChange[mid],
                    "inserted",
                    estimatedWidth
                ));
            }
        }
        return SortExtrema(result);
    }

    private static List<Extremum> MergeTriplets(IReadOnlyList<Extremum> extrema, int thresholdFrames, double[] rawChange, double[] processedChange)
    {
        if (extrema.Count < 3)
        {
            return new List<Extremum>(extrema);
        }
        var merged = new List<Extremum>();
        int i = 0;
        while (i < extrema.Count)
        {
            if (i + 2 < extrema.Count)
            {
                var a = extrema[i];
                var b = extrema[i + 1];
                var c = extrema[i + 2];
                if (a.Kind == c.Kind && a.Kind != b.Kind && (c.Frame - a.Frame) <= thresholdFrames)
                {
                    int midFrame = (int)Math.Clamp(Math.Round((a.Frame + c.Frame) / 2.0), 0, rawChange.Length - 1);
                    merged.Add(new Extremum(
                        midFrame,
                        a.Kind,
                        rawChange[midFrame],
                        processedChange[midFrame],
                        Math.Max(a.Prominence, c.Prominence),
                        Math.Min(a.WindowMin, c.WindowMin),
                        Math.Max(a.WindowMax, c.WindowMax),
                        "merged",
                        Math.Max(a.Width, c.Width)
                    ));
                    i += 3;
                    continue;
                }
            }
            merged.Add(extrema[i]);
            i++;
        }
        return merged;
    }

    private static List<Extremum> SortExtrema(IEnumerable<Extremum> extrema)
    {
        return extrema
            .OrderBy(e => e.Frame)
            .ThenBy(e => e.Kind == "peak" ? 0 : 1)
            .ToList();
    }

    private static List<GraphPoint> ApplySlopeConstraints(IReadOnlyList<Extremum> extrema, double minSlope, double maxSlope, double boostSlope)
    {
        if (extrema.Count == 0)
        {
            return new List<GraphPoint>();
        }

        minSlope = Math.Max(0.0, minSlope);
        maxSlope = maxSlope <= 0.0 ? double.PositiveInfinity : Math.Max(maxSlope, minSlope);

        var graph = new List<GraphPoint>();
        var current = extrema[0];
        double currentX = current.Frame;
        double currentY = MakeTargetY(current);
        graph.Add(new GraphPoint(currentX, currentY, current.Kind, current.Origin));

        const double offsetDecay = 0.82;
        const double offsetClamp = 15.0;
        double currentOffset = 0.0;

        int i = 1;
        while (i < extrema.Count)
        {
            var target = extrema[i];
            double targetY = MakeTargetY(target);

            double baseDx = target.Frame - currentX;
            double baseDy = targetY - currentY;
            double baseSlope = baseDx > 1e-9 ? Math.Abs(baseDy) / baseDx : double.PositiveInfinity;
            double desiredOffset = ComputeSlopeOffset(baseSlope);

            currentOffset = currentOffset * offsetDecay + desiredOffset * (1.0 - offsetDecay);
            currentOffset = Math.Clamp(currentOffset, 0.0, offsetClamp);

            double candidateX = target.Frame + currentOffset;
            if (i + 1 < extrema.Count)
            {
                double nextFrame = extrema[i + 1].Frame;
                candidateX = Math.Min(candidateX, nextFrame - 0.5);
            }
            candidateX = Math.Max(candidateX, target.Frame);
            candidateX = Math.Max(candidateX, currentX + 1e-4);

            double targetX = candidateX;
            double dx = targetX - currentX;
            if (dx <= 0)
            {
                i++;
                continue;
            }

            double dy = targetY - currentY;
            double actualSlope = Math.Abs(dy) / dx;
            string direction = dy > 0 ? "to_peak" : "to_trough";

            if (actualSlope > maxSlope)
            {
                double maxDelta = maxSlope * dx;
                double reachable = currentY + Math.Sign(dy) * maxDelta;
                reachable = Clamp(reachable, TroughValue, PeakValue);
                graph.Add(new GraphPoint(targetX, reachable, "intermediate", "max_slope", direction));
                currentX = targetX;
                currentY = reachable;
                i++;
                continue;
            }

            if (actualSlope < minSlope && Math.Abs(dy) > 1e-9)
            {
                var (points, nextX, nextY) = HandleMinSlopeSegment(currentX, currentY, targetX, targetY, direction, target.Kind, target.Origin, minSlope, maxSlope);
                graph.AddRange(points);
                currentX = nextX;
                currentY = nextY;
                i++;
                continue;
            }

            graph.Add(new GraphPoint(targetX, targetY, target.Kind, target.Origin, direction));
            currentX = targetX;
            currentY = targetY;
            i++;
        }

        return graph;
    }

    private static double ComputeSlopeOffset(double slope)
    {
        if (!double.IsFinite(slope))
        {
            slope = 100.0;
        }

        const double minOffset = 1.0;
        const double maxOffset = 15.0;
        const double gentleSlope = 2.0;
        const double steepSlope = 12.0;

        if (slope <= gentleSlope)
        {
            return maxOffset;
        }

        if (slope >= steepSlope)
        {
            return minOffset;
        }

        double normalized = (slope - gentleSlope) / (steepSlope - gentleSlope);
        double offset = maxOffset - normalized * (maxOffset - minOffset);
        return Math.Clamp(offset, minOffset, maxOffset);
    }

    private static double MakeTargetY(Extremum extremum)
    {
        double min = extremum.WindowMin;
        double max = extremum.WindowMax;
        double normalized = max > min ? (extremum.RawChange - min) / (max - min) : 0.5;
        double halfRange = (PeakValue - TroughValue) / 2.0;
        return extremum.Kind == "peak"
            ? PeakValue - halfRange * (1.0 - normalized)
            : TroughValue + halfRange * normalized;
    }
    private static (IReadOnlyList<GraphPoint> Points, double NextX, double NextY) HandleMinSlopeSegment(
        double currentX,
        double currentY,
        double targetX,
        double targetY,
        string direction,
        string targetKind,
        string targetOrigin,
        double minSlope,
        double maxSlope)
    {
        double dx = targetX - currentX;
        double dy = targetY - currentY;
        if (dx <= 0)
        {
            return (Array.Empty<GraphPoint>(), currentX, currentY);
        }

        double minDirectDx = minSlope > 0 ? Math.Abs(dy) / minSlope : dx;
        double extraDx = dx - minDirectDx;
        if (extraDx <= 0)
        {
            return (new[] { new GraphPoint(targetX, targetY, targetKind, targetOrigin, direction) }, targetX, targetY);
        }

        double fullAmplitude = PeakValue - TroughValue;
        double halfDxMin = minSlope > 0 ? fullAmplitude / minSlope : 0.0;
        double halfDxMax = (maxSlope > 0 && !double.IsInfinity(maxSlope)) ? fullAmplitude / maxSlope : 0.0;
        double halfDx = Math.Max(Math.Max(halfDxMin, halfDxMax), 1e-6);
        int maxTransitions = (int)(extraDx / halfDx);
        bool isAtExtreme = Math.Abs(currentY - PeakValue) < 1e-6 || Math.Abs(currentY - TroughValue) < 1e-6;
        int numTransitions = isAtExtreme ? (maxTransitions / 2) * 2 : maxTransitions;
        if (!isAtExtreme && numTransitions <= 0)
        {
            numTransitions = 1;
        }

        var points = new List<GraphPoint>();
        if (numTransitions <= 0)
        {
            points.Add(new GraphPoint(targetX, targetY, targetKind, targetOrigin, direction));
            return (points, targetX, targetY);
        }

        int totalSegments = numTransitions + 1;
        double segmentDx = dx / totalSegments;
        double currentPos = currentX;
        double prevValue = currentY;

        var sequence = new List<double>();
        if (isAtExtreme)
        {
            for (int i = 0; i < numTransitions; i++)
            {
                sequence.Add(i % 2 == 0 ? (direction == "to_peak" ? PeakValue : TroughValue) : (direction == "to_peak" ? TroughValue : PeakValue));
            }
            sequence.Add(targetY);
        }
        else
        {
            bool goingUp = direction == "to_peak";
            for (int i = 0; i < numTransitions; i++)
            {
                double value = goingUp ? PeakValue : TroughValue;
                sequence.Add(value);
                goingUp = !goingUp;
            }
            sequence.Add(targetY);
        }

        foreach (double value in sequence)
        {
            currentPos += segmentDx;
            currentPos = Math.Min(currentPos, targetX);
            double clamped = Clamp(value, TroughValue, PeakValue);
            points.Add(new GraphPoint(currentPos, clamped, "intermediate", "min_slope", direction));
            prevValue = clamped;
        }

        if (points.Count > 0)
        {
            var last = points[^1];
            points[^1] = new GraphPoint(last.Position, last.Value, targetKind, targetOrigin, direction);
        }

        var lastPoint = points.Count > 0 ? points[^1] : new GraphPoint(targetX, targetY, targetKind, targetOrigin, direction);
        return (points, lastPoint.Position, lastPoint.Value);
    }


    private static List<GraphPoint> BoostFlatPeaks(IReadOnlyList<GraphPoint> points, double boostSlope)
    {
        if (points.Count < 3)
        {
            return new List<GraphPoint>(points);
        }

        double peakThreshold = PeakValue * 0.75;
        var boosted = new List<GraphPoint>();
        for (int i = 0; i < points.Count; i++)
        {
            var current = points[i];
            bool hasPrev = i > 0;
            bool hasNext = i + 1 < points.Count;
            if (!hasPrev || !hasNext)
            {
                boosted.Add(current);
                continue;
            }

            var prev = points[i - 1];
            var next = points[i + 1];
            if (prev.Label != "trough" || next.Label != "trough" || current.Label != "peak" || current.Value < peakThreshold)
            {
                boosted.Add(current);
                continue;
            }

            double leftDx = current.Position - prev.Position;
            double rightDx = next.Position - current.Position;
            if (leftDx <= 0 || rightDx <= 0)
            {
                boosted.Add(current);
                continue;
            }

            double leftSlope = Math.Abs(current.Value - prev.Value) / leftDx;
            double rightSlope = Math.Abs(next.Value - current.Value) / rightDx;
            if (leftSlope > boostSlope || rightSlope > boostSlope)
            {
                boosted.Add(current);
                continue;
            }

            double leftX = current.Position - leftDx * 0.4;
            double rightX = current.Position + rightDx * 0.4;
            if (leftX <= prev.Position || rightX >= next.Position)
            {
                boosted.Add(current);
                continue;
            }

            double neighborAvg = (prev.Value + next.Value) / 2.0;
            double peakToNeighbor = current.Value - neighborAvg;
            if (peakToNeighbor <= 0)
            {
                boosted.Add(current);
                continue;
            }

            double shoulderY = current.Value - peakToNeighbor * 0.2;
            if (shoulderY >= current.Value || shoulderY <= neighborAvg)
            {
                boosted.Add(current);
                continue;
            }

            boosted.Add(new GraphPoint(leftX, shoulderY, "boosted", "boosted"));
            boosted.Add(current);
            boosted.Add(new GraphPoint(rightX, shoulderY, "boosted", "boosted"));
        }

        return boosted;
    }

    private static List<GraphPoint> ApplyCentralDeviationConstraint(IReadOnlyList<GraphPoint> points, IReadOnlyList<Extremum> extrema, double thresholdRatio)
    {
        if (points.Count == 0 || extrema.Count == 0 || thresholdRatio <= 0)
        {
            return points.ToList();
        }

        double globalMin = extrema.Min(e => e.WindowMin);
        double globalMax = extrema.Max(e => e.WindowMax);
        double globalRange = globalMax - globalMin;
        if (globalRange <= 1e-9)
        {
            return points.ToList();
        }

        double ratioLimit = Math.Max(0.0, thresholdRatio);
        const double dampingFactor = 0.3;

        var lookup = extrema
            .GroupBy(e => Math.Round(e.Frame, 3))
            .ToDictionary(g => g.Key, g => g.OrderByDescending(e => e.Prominence).First());

        var adjusted = new List<GraphPoint>(points.Count);
        foreach (var point in points)
        {
            double key = Math.Round(point.Position, 3);
            if (lookup.TryGetValue(key, out var extremum))
            {
                double windowRange = extremum.WindowMax - extremum.WindowMin;
                if (windowRange > 0)
                {
                    double ratio = windowRange / globalRange;
                    if (ratio < ratioLimit)
                    {
                        double center = (PeakValue + TroughValue) * 0.5;
                        double newValue = center + (point.Value - center) * dampingFactor;
                        string origin = string.IsNullOrEmpty(point.Origin) ? "central_adjusted" : point.Origin.Contains("central_adjusted") ? point.Origin : point.Origin + "|central_adjusted";
                        adjusted.Add(new GraphPoint(point.Position, Clamp(newValue, TroughValue, PeakValue), point.Label, origin, point.Direction));
                        continue;
                    }
                }
            }

            adjusted.Add(point);
        }

        return adjusted;
    }

    private static List<GraphPoint> DedupeGraphPoints(IEnumerable<GraphPoint> points)
    {
        var cleaned = new List<GraphPoint>();
        double? last = null;
        foreach (var pt in points)
        {
            double pos = pt.Position;
            if (last.HasValue && pos <= last.Value)
            {
                pos = last.Value + 1e-6;
            }
            cleaned.Add(new GraphPoint(pos, Clamp(pt.Value, TroughValue, PeakValue), pt.Label, pt.Origin, pt.Direction));
            last = pos;
        }
        return cleaned;
    }

    private static double[] InterpolateGraph(IReadOnlyList<GraphPoint> points, int n)
    {
        double[] values = new double[n];
        if (points.Count == 0)
        {
            Array.Fill(values, 50.0);
            return values;
        }

        if (points.Count == 1)
        {
            Array.Fill(values, points[0].Value);
            return values;
        }

        double[] positions = points.Select(p => p.Position).ToArray();
        double[] graphValues = points.Select(p => p.Value).ToArray();

        int seg = 0;
        for (int i = 0; i < n; i++)
        {
            double x = i;
            while (seg + 1 < positions.Length && x > positions[seg + 1])
            {
                seg++;
            }

            if (seg >= positions.Length - 1)
            {
                values[i] = graphValues[^1];
            }
            else
            {
                double x0 = positions[seg];
                double x1 = positions[seg + 1];
                double y0 = graphValues[seg];
                double y1 = graphValues[seg + 1];
                if (Math.Abs(x1 - x0) < 1e-9)
                {
                    values[i] = y0;
                }
                else
                {
                    double t = (x - x0) / (x1 - x0);
                    values[i] = y0 + (y1 - y0) * t;
                }
            }
        }

        return values;
    }

    private static int MsToFrames(double durationMs, double frameRate)
    {
        if (durationMs <= 0 || frameRate <= 0)
        {
            return 0;
        }
        return Math.Max(1, (int)Math.Round(durationMs / 1000.0 * frameRate));
    }

    private static double Clamp(double value, double min, double max) => value < min ? min : value > max ? max : value;

    private static double[] ApplyFftDenoise(double[] signal, int framesPerComponent, int? windowFrames)
    {
        int length = signal.Length;
        if (length == 0)
        {
            return Array.Empty<double>();
        }

        framesPerComponent = Math.Max(1, framesPerComponent);
        int? window = windowFrames.HasValue && windowFrames.Value > 0 ? windowFrames.Value : null;
        if (window is null || window.Value >= length)
        {
            return FilterSegment(signal, framesPerComponent);
        }

        int win = Math.Min(Math.Max(1, window.Value), length);
        double[] output = new double[length];
        int start = 0;
        while (start < length)
        {
            int segmentLength = Math.Min(win, length - start);
            double[] segment = new double[segmentLength];
            Array.Copy(signal, start, segment, 0, segmentLength);
            double[] filtered = FilterSegment(segment, framesPerComponent);
            Array.Copy(filtered, 0, output, start, segmentLength);
            start += segmentLength;
        }
        return output;

        static double[] FilterSegment(double[] segment, int framesPerComponent)
        {
            int len = segment.Length;
            if (len == 0)
            {
                return Array.Empty<double>();
            }

            int keep = Math.Max(1, len / framesPerComponent);
            if (keep * 2 >= len)
            {
                return (double[])segment.Clone();
            }

            var spectrum = new Complex[len];
            for (int i = 0; i < len; i++)
            {
                spectrum[i] = new Complex(segment[i], 0.0);
            }

            Fourier.Forward(spectrum, FourierOptions.Matlab);
            int upper = len - keep;
            for (int i = keep; i < upper; i++)
            {
                spectrum[i] = Complex.Zero;
            }
            Fourier.Inverse(spectrum, FourierOptions.Matlab);

            double[] filtered = new double[len];
            for (int i = 0; i < len; i++)
            {
                filtered[i] = spectrum[i].Real;
            }
            return filtered;
        }
    }


    private readonly record struct Extremum(double Frame, string Kind, double RawChange, double ProcessedChange, double Prominence, double WindowMin, double WindowMax, string Origin, double Width);

    private readonly record struct GraphPoint(double Position, double Value, string Label, string Origin, string? Direction = null);
}

public readonly record struct ExtremumInfo(double Frame, string Kind, string Origin, double RawChange, double Prominence);

public readonly record struct GraphPointInfo(double Position, double Value, string Label, string Origin, string? Direction);

public readonly record struct DelayMarkerInfo(double Frame, double Delay);

public sealed record PostprocessResult(
    double[] RawSignal,
    double[] SmoothedSignal,
    double[] DenoisedSignal,
    IReadOnlyList<ExtremumInfo> RawExtrema,
    IReadOnlyList<ExtremumInfo> StageTwoExtrema,
    IReadOnlyList<GraphPointInfo> GraphPoints,
    IReadOnlyList<DelayMarkerInfo> TemporalDelayMarkers,
    double[] TemporalDelayProfile,
    double[] ProcessedValue,
    double[] ProcessedChange,
    double[] PhaseMarker,
    string[] PhaseSource,
    double[] RawExtremaMarker,
    int TemporalShiftFrames,
    PostprocessOptions Options)
{
    public static PostprocessResult Fallback(int length, PostprocessOptions options)
    {
        double[] processedValue = Enumerable.Repeat(50.0, length).ToArray();
        double[] processedChange = new double[length];
        double[] phaseMarker = Enumerable.Repeat(double.NaN, length).ToArray();
        string[] phaseSource = Enumerable.Repeat(string.Empty, length).ToArray();
        double[] rawMarker = Enumerable.Repeat(double.NaN, length).ToArray();
        return new PostprocessResult(
            Array.Empty<double>(),
            Array.Empty<double>(),
            Array.Empty<double>(),
            Array.Empty<ExtremumInfo>(),
            Array.Empty<ExtremumInfo>(),
            Array.Empty<GraphPointInfo>(),
            Array.Empty<DelayMarkerInfo>(),
            Array.Empty<double>(),
            processedValue,
            processedChange,
            phaseMarker,
            phaseSource,
            rawMarker,
            0,
            options
        );
    }
}

























