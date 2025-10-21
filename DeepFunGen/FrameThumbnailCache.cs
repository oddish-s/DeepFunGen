namespace app;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Extensions;

internal sealed class FrameThumbnailCache : IDisposable
{
    private readonly string _videoPath;
    private readonly int _targetWidth;
    private readonly object _cacheLock = new();
    private readonly Dictionary<int, Bitmap> _cache = new();
    private readonly LinkedList<int> _lruOrder = new();
    private readonly ConcurrentDictionary<int, Task<Bitmap?>> _loadTasks = new();
    private readonly int _maxEntries;
    private bool _disposed;

    public FrameThumbnailCache(string videoPath, int targetWidth = 256, int maxEntries = 60)
    {
        if (string.IsNullOrWhiteSpace(videoPath))
        {
            throw new ArgumentException("Video path must be provided", nameof(videoPath));
        }

        if (!File.Exists(videoPath))
        {
            throw new FileNotFoundException("Video file not found", videoPath);
        }

        _videoPath = videoPath;
        _targetWidth = Math.Clamp(targetWidth, 64, 512);
        _maxEntries = Math.Max(10, maxEntries);
    }

    public Task<Bitmap?> GetThumbnailAsync(int frameIndex, CancellationToken cancellationToken)
    {
        if (frameIndex < 0)
        {
            return Task.FromResult<Bitmap?>(null);
        }

        Bitmap? cached = TryGetFromCache(frameIndex);
        if (cached is not null)
        {
            return Task.FromResult<Bitmap?>(cached);
        }

        Task<Bitmap?> loader = _loadTasks.GetOrAdd(frameIndex, idx => Task.Run(() => LoadThumbnail(idx, cancellationToken), cancellationToken));
        return loader.ContinueWith(task =>
        {
            _loadTasks.TryRemove(frameIndex, out _);
            if (task.IsCanceled || cancellationToken.IsCancellationRequested)
            {
                return null;
            }

            if (task.IsFaulted)
            {
                return null;
            }

            Bitmap? bitmap = task.Result;
            if (bitmap is null)
            {
                return null;
            }

            AddToCache(frameIndex, bitmap);
            return CloneBitmap(bitmap);
        }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
    }

    private Bitmap? TryGetFromCache(int frameIndex)
    {
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(frameIndex, out Bitmap? bitmap))
            {
                MoveToMostRecent(frameIndex);
                return CloneBitmap(bitmap);
            }
        }

        return null;
    }

    private void AddToCache(int frameIndex, Bitmap bitmap)
    {
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(frameIndex, out Bitmap? existing))
            {
                existing.Dispose();
                _cache[frameIndex] = bitmap;
                MoveToMostRecent(frameIndex);
                return;
            }

            _cache[frameIndex] = bitmap;
            _lruOrder.AddFirst(frameIndex);
            TrimCacheIfNeeded();
        }
    }

    private void TrimCacheIfNeeded()
    {
        while (_lruOrder.Count > _maxEntries)
        {
            int frameToRemove = _lruOrder.Last!.Value;
            _lruOrder.RemoveLast();
            if (_cache.Remove(frameToRemove, out Bitmap? bitmap))
            {
                bitmap.Dispose();
            }
        }
    }

    private void MoveToMostRecent(int frameIndex)
    {
        LinkedListNode<int>? node = _lruOrder.Find(frameIndex);
        if (node is null)
        {
            return;
        }

        _lruOrder.Remove(node);
        _lruOrder.AddFirst(node);
    }

    private Bitmap? LoadThumbnail(int frameIndex, CancellationToken cancellationToken)
    {
        try
        {
            using var capture = new VideoCapture(_videoPath);
            if (!capture.IsOpened())
            {
                return null;
            }

            capture.Set(VideoCaptureProperties.PosFrames, frameIndex);
            using var frame = new Mat();
            if (!capture.Read(frame) || frame.Empty())
            {
                return null;
            }

            if (cancellationToken.IsCancellationRequested)
            {
                return null;
            }

            int width = frame.Width;
            int height = frame.Height;
            if (width == 0 || height == 0)
            {
                return null;
            }

            double scale = (double)_targetWidth / width;
            if (scale <= 0.0 || scale >= 1.0)
            {
                scale = Math.Min(1.0, _targetWidth / (double)Math.Max(width, 1));
            }

            int targetWidth = Math.Max(32, (int)Math.Round(width * scale));
            int targetHeight = Math.Max(32, (int)Math.Round(height * scale));
            using var resized = new Mat();
            Cv2.Resize(frame, resized, new OpenCvSharp.Size(targetWidth, targetHeight), 0, 0, InterpolationFlags.Area);

            if (cancellationToken.IsCancellationRequested)
            {
                return null;
            }

            return BitmapConverter.ToBitmap(resized);
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static Bitmap CloneBitmap(Bitmap source)
    {
        return source.Clone(new Rectangle(0, 0, source.Width, source.Height), source.PixelFormat);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        lock (_cacheLock)
        {
            foreach (Bitmap bitmap in _cache.Values)
            {
                bitmap.Dispose();
            }

            _cache.Clear();
            _lruOrder.Clear();
        }
    }
}
