namespace app;

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

public enum VideoJobStatus
{
    Pending,
    Processing,
    Skipped,
    Completed,
    Failed,
}

public sealed class VideoJob : INotifyPropertyChanged
{
    private VideoJobStatus _status;
    private string _message = string.Empty;
    private double _progress = double.NaN;
    private string? _predictionPath;

    public VideoJob(string videoPath)
    {
        VideoPath = videoPath;
        Status = VideoJobStatus.Pending;
    }

    public string VideoPath { get; }

    public string? PredictionPath
    {
        get => _predictionPath;
        set => SetField(ref _predictionPath, value);
    }

    public VideoJobStatus Status
    {
        get => _status;
        set => SetField(ref _status, value);
    }

    public string Message
    {
        get => _message;
        set => SetField(ref _message, value);
    }


    public VideoProcessingService.PredictionSummary? LatestSummary { get; set; }
    public double Progress
    {
        get => _progress;
        set
        {
            double clamped = double.IsNaN(value) ? double.NaN : Math.Clamp(value, 0d, 1d);
            if (double.IsNaN(_progress) && double.IsNaN(clamped))
            {
                return;
            }
            if (!double.IsNaN(_progress) && !double.IsNaN(clamped) && Math.Abs(_progress - clamped) < 1e-6)
            {
                return;
            }

            _progress = clamped;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ProgressText));
        }
    }

    public string ProgressText => double.IsNaN(_progress) ? "-" : _progress.ToString("P0");

    public event PropertyChangedEventHandler? PropertyChanged;

    private void SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        if (propertyName != null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        if (propertyName != null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
