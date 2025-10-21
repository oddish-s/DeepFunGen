namespace app;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

public sealed class PredictionViewerForm : Form
{
    private readonly PlotModel _model;
    private readonly PostprocessOptions _options;
    private readonly PredictionCanvas _canvas;
    private readonly TrackBar _windowTrack;
    private readonly Label _windowLabel;
    private readonly PictureBox _previewPicture;
    private readonly Label _previewLabel;
    private readonly Label _previewInfo;
    private readonly Label _valueLabel;
    private readonly VerticalProgressBar _valueBar;
    private readonly Panel _previewPanel;
    private readonly FrameThumbnailCache? _thumbnailCache;
    private readonly double[] _timestamps;
    private CancellationTokenSource? _previewCts;
    private int _lastPreviewFrame = -1;

    public PredictionViewerForm(VideoProcessingService.PredictionSummary summary)
    {
        if (summary is null)
        {
            throw new ArgumentNullException(nameof(summary));
        }

        _options = summary.Options ?? PostprocessOptions.Default;
        _model = BuildModel(summary);
        _timestamps = summary.Timestamps ?? Array.Empty<double>();
        _thumbnailCache = TryCreateThumbnailCache(summary.Job.VideoPath);

        Text = $"{Path.GetFileName(summary.Job.VideoPath)} Changes";
        StartPosition = FormStartPosition.CenterParent;
        Size = new Size(1520, 800);
        MinimumSize = new Size(1280, 680);

        _canvas = new PredictionCanvas(_model)
        {
            Dock = DockStyle.Fill,
            BackColor = Color.White,
        };
        _canvas.HoveredFrameChanged += HandleHoveredFrameChanged;

        _windowLabel = new Label
        {
            Text = "Window: -",
            AutoSize = true,
            Padding = new Padding(10, 8, 10, 8),
            Dock = DockStyle.Fill,
        };

        _windowTrack = new TrackBar
        {
            Minimum = 0,
            TickStyle = TickStyle.None,
            Dock = DockStyle.Fill,
            SmallChange = 5,
            LargeChange = 25,
        };
        _windowTrack.ValueChanged += (_, _) => ApplyWindowFromTrack();

        int maxOffset = Math.Max(0, _model.VisibleCount - PredictionCanvas.WindowSize);
        _windowTrack.Maximum = Math.Max(_windowTrack.Minimum, maxOffset);
        _windowTrack.Enabled = maxOffset > 0;
        if (!_windowTrack.Enabled)
        {
            _windowTrack.Value = 0;
        }

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
        };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        _previewPicture = new PictureBox
        {
            Dock = DockStyle.Fill,
            BackColor = Color.Black,
            SizeMode = PictureBoxSizeMode.Zoom,
            BorderStyle = BorderStyle.FixedSingle,
        };

        _previewLabel = new Label
        {
            Dock = DockStyle.Top,
            AutoSize = false,
            Height = 24,
            Padding = new Padding(4, 4, 4, 0),
            Font = new Font(SystemFonts.MessageBoxFont.FontFamily, SystemFonts.MessageBoxFont.Size, FontStyle.Bold),
            Text = "Preview",
        };

        _previewInfo = new Label
        {
            Dock = DockStyle.Top,
            AutoSize = false,
            Height = 32,
            Padding = new Padding(4, 0, 4, 6),
            TextAlign = ContentAlignment.MiddleLeft,
            Text = "Hover stage 3 graph to preview",
        };

        _valueLabel = new Label
        {
            Dock = DockStyle.Top,
            AutoSize = false,
            Height = 22,
            Padding = new Padding(4, 0, 4, 4),
            TextAlign = ContentAlignment.MiddleLeft,
            Text = "Value: -",
        };

        _valueBar = new VerticalProgressBar
        {
            Dock = DockStyle.Fill,
            Minimum = 0,
            Maximum = 100,
            MaximumSize = new Size(20, 9999),
            Value = 0,
            Margin = new Padding(4, 4, 4, 8),
            BackColor = Color.FromArgb(236, 239, 244),
            ForeColor = Color.SteelBlue,
        };

        var valuePanel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 2,
            Padding = new Padding(6, 4, 6, 10),
            MinimumSize = new Size(0, 160),
        };
        valuePanel.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        valuePanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
        valuePanel.Controls.Add(_valueLabel, 0, 0);
        valuePanel.Controls.Add(_valueBar, 0, 1);

        var imageValueLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 2,
        };
        imageValueLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 68f));
        imageValueLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 32f));
        imageValueLayout.Controls.Add(_previewPicture, 0, 0);
        imageValueLayout.Controls.Add(valuePanel, 0, 1);

        var previewLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
        };
        previewLayout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        previewLayout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        previewLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
        previewLayout.Controls.Add(_previewLabel, 0, 0);
        previewLayout.Controls.Add(_previewInfo, 0, 1);
        previewLayout.Controls.Add(imageValueLayout, 0, 2);

        _previewPanel = new Panel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(8, 0, 8, 0),
            MinimumSize = new Size(440, 0),
        };
        _previewPanel.Controls.Add(previewLayout);

        var optionsLabel = new Label
        {
            Text = $"Options: {_options} | shift={_model.TemporalShiftFrames} frames",
            Dock = DockStyle.Fill,
            AutoSize = true,
            Padding = new Padding(10, 8, 10, 8),
        };
        layout.Controls.Add(optionsLabel, 0, 0);

        var centerLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
        };
        centerLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
        centerLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 440f));
        centerLayout.Controls.Add(_canvas, 0, 0);
        centerLayout.Controls.Add(_previewPanel, 1, 0);
        layout.Controls.Add(centerLayout, 0, 1);

        var scrollPanel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            AutoSize = true,
        };
        scrollPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        scrollPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
        scrollPanel.Controls.Add(_windowLabel, 0, 0);
        scrollPanel.Controls.Add(_windowTrack, 1, 0);
        layout.Controls.Add(scrollPanel, 0, 2);

        Controls.Add(layout);

        FormClosed += (_, _) => DisposePreview();
        ApplyWindowFromTrack();
    }

    private void ApplyWindowFromTrack()
    {
        int offset = _windowTrack.Enabled ? _windowTrack.Value : 0;
        _canvas.SetWindowStart(offset);
        (int startFrame, int endFrame) = _canvas.GetWindowFrameRange();
        _windowLabel.Text = startFrame <= endFrame ? $"Window: {startFrame} - {endFrame}" : "Window: -";
        ResetPreview();
    }

    private FrameThumbnailCache? TryCreateThumbnailCache(string videoPath)
    {
        try
        {
            return File.Exists(videoPath) ? new FrameThumbnailCache(videoPath, 240, 80) : null;
        }
        catch
        {
            return null;
        }
    }

    private void HandleHoveredFrameChanged(object? sender, PredictionCanvas.FrameEventArgs e)
    {
        if (IsDisposed)
        {
            return;
        }

        if (InvokeRequired)
        {
            BeginInvoke(new Action<object?, PredictionCanvas.FrameEventArgs>(HandleHoveredFrameChanged), sender, e);
            return;
        }

        _ = UpdatePreviewAsync(e.FrameIndex, e.ProcessedValue);
    }

    private async Task UpdatePreviewAsync(int frameIndex, double? processedValue)
    {
        UpdateValueOverlay(processedValue);
        if (_thumbnailCache is null)
        {
            ShowPreviewUnavailable("Preview unavailable (video file missing).");
            return;
        }

        if (frameIndex < 0)
        {
            ResetPreview();
            return;
        }

        if (frameIndex == _lastPreviewFrame)
        {
            return;
        }

        _lastPreviewFrame = frameIndex;
        var cts = new CancellationTokenSource();
        var previous = Interlocked.Exchange(ref _previewCts, cts);
        previous?.Cancel();
        previous?.Dispose();
        var token = cts.Token;

        _previewInfo.Text = $"Loading frame {frameIndex}...";

        try
        {
            var bitmap = await _thumbnailCache.GetThumbnailAsync(frameIndex, token).ConfigureAwait(true);
            if (token.IsCancellationRequested)
            {
                bitmap?.Dispose();
                return;
            }

            if (bitmap is null)
            {
                ShowPreviewUnavailable("Frame not available.");
                return;
            }

            ReplacePreviewImage(bitmap);
            _previewInfo.Text = FormatPreviewCaption(frameIndex);
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
            ShowPreviewUnavailable("Failed to load frame.");
        }
    }

    private void ReplacePreviewImage(Image image)
    {
        var previous = _previewPicture.Image;
        _previewPicture.Image = image;
        previous?.Dispose();
    }

    private void UpdateValueOverlay(double? processedValue)
    {
        if (processedValue.HasValue)
        {
            double clamped = Math.Clamp(processedValue.Value, 0.0, 100.0);
            _valueLabel.Text = $"Value: {clamped:0.###}";
            int intValue = (int)Math.Round(clamped);
            intValue = Math.Min(_valueBar.Maximum, Math.Max(_valueBar.Minimum, intValue));
            if (_valueBar.Value != intValue)
            {
                _valueBar.Value = intValue;
            }
            else
            {
                _valueBar.Refresh();
            }
        }
        else
        {
            _valueLabel.Text = "Value: -";
            _valueBar.Value = _valueBar.Minimum;
        }
    }

    private string FormatPreviewCaption(int frameIndex)
    {
        if (_timestamps.Length == 0)
        {
            return $"Frame {frameIndex}";
        }

        double timestampMs;
        if (frameIndex < _timestamps.Length)
        {
            timestampMs = _timestamps[frameIndex];
        }
        else
        {
            double delta = _timestamps.Length > 1 ? Math.Max(1.0, _timestamps[1] - _timestamps[0]) : 33.333;
            timestampMs = _timestamps[^1] + (frameIndex - (_timestamps.Length - 1)) * delta;
        }

        double seconds = timestampMs / 1000.0;
        return $"Frame {frameIndex} | {seconds:0.###} s";
    }

    private void ShowPreviewUnavailable(string message)
    {
        _previewInfo.Text = message;
    }

    private void ResetPreview()
    {
        _lastPreviewFrame = -1;
        var current = Interlocked.Exchange(ref _previewCts, null);
        current?.Cancel();
        current?.Dispose();
        _previewInfo.Text = _thumbnailCache is null
            ? "Preview unavailable (video file missing)."
            : "Hover stage 3 graph to preview";
        UpdateValueOverlay(null);
    }

    private void DisposePreview()
    {
        var current = Interlocked.Exchange(ref _previewCts, null);
        current?.Cancel();
        current?.Dispose();
        _thumbnailCache?.Dispose();
        UpdateValueOverlay(null);
        if (_previewPicture.Image is Image image)
        {
            _previewPicture.Image = null;
            image.Dispose();
        }
    }

    private static PlotModel BuildModel(VideoProcessingService.PredictionSummary summary)
    {
        int frameCount = summary.PredictedChanges.Length;
        int skip = Math.Max(0, OnnxSequenceModel.SequenceLength - 1);
        if (frameCount == 0)
        {
            return new PlotModel(
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<ExtremumInfo>(),
                Array.Empty<ExtremumInfo>(),
                Array.Empty<GraphPointInfo>(),
                Array.Empty<DelayMarkerInfo>(),
                Array.Empty<double>(),
                0,
                skip);
        }

        var frames = Enumerable.Range(0, frameCount).Select(i => (double)i).ToArray();
        var rawChange = ToDouble(summary.PredictedChanges);
        var result = summary.DetailedResult;

        var rawSignal = result.RawSignal.Length > 0 ? result.RawSignal : rawChange;
        var smoothed = result.SmoothedSignal.Length > 0 ? result.SmoothedSignal : rawSignal;
        var denoised = result.DenoisedSignal.Length > 0 ? result.DenoisedSignal : Array.Empty<double>();
        int temporalShift = result.TemporalShiftFrames;
        var processedValue = result.ProcessedValue.Length > 0
            ? result.ProcessedValue
            : ToDouble(summary.ProcessedValues);
        var processedChange = result.ProcessedChange.Length > 0
            ? result.ProcessedChange
            : ToDouble(summary.ProcessedChanges);
        var phaseMarker = result.PhaseMarker.Length > 0
            ? result.PhaseMarker
            : Array.ConvertAll(summary.PhaseMarker, v => (double)v);

        var rawExtrema = result.RawExtrema.Where(e => e.Frame >= 0 && e.Frame < frameCount).ToArray();
        var stageExtrema = result.StageTwoExtrema.Where(e => e.Frame >= 0 && e.Frame < frameCount).ToArray();
        var graphPoints = result.GraphPoints.Where(g => g.Position >= 0 && g.Position < frameCount).ToArray();
        var delayMarkers = result.TemporalDelayMarkers.Count > 0
            ? result.TemporalDelayMarkers.ToArray()
            : Array.Empty<DelayMarkerInfo>();
        var delayProfile = result.TemporalDelayProfile.Length > 0
            ? result.TemporalDelayProfile
            : Array.Empty<double>();

        return new PlotModel(
            frames,
            rawChange,
            rawSignal,
            denoised,
            smoothed,
            processedChange,
            processedValue,
            phaseMarker,
            rawExtrema,
            stageExtrema,
            graphPoints,
            delayMarkers,
            delayProfile,
            temporalShift,
            skip);
    }

    private static double[] ToDouble(float[] source) => Array.ConvertAll(source, static v => (double)v);

    private sealed class PlotModel
    {
        public PlotModel(
            double[] frames,
            double[] rawChange,
            double[] rawSignal,
            double[] denoised,
            double[] smoothed,
            double[] processedChange,
            double[] processedValue,
            double[] phaseMarker,
            ExtremumInfo[] rawExtrema,
            ExtremumInfo[] stageTwoExtrema,
            GraphPointInfo[] graphPoints,
            DelayMarkerInfo[] delayMarkers,
            double[] delayProfile,
            int temporalShiftFrames,
            int skip)
        {
            Frames = frames;
            RawChange = rawChange;
            RawSignal = rawSignal;
            Denoised = denoised;
            Smoothed = smoothed;
            ProcessedChange = processedChange;
            ProcessedValue = processedValue;
            PhaseMarker = phaseMarker;
            RawExtrema = rawExtrema;
            StageTwoExtrema = stageTwoExtrema;
            GraphPoints = graphPoints;
            DelayMarkers = delayMarkers;
            DelayProfile = delayProfile;
            TemporalShiftFrames = temporalShiftFrames;
            Skip = skip;
        }

        public double[] Frames { get; }
        public double[] RawChange { get; }
        public double[] RawSignal { get; }
        public double[] Denoised { get; }
        public double[] Smoothed { get; }
        public double[] ProcessedChange { get; }
        public double[] ProcessedValue { get; }
        public double[] PhaseMarker { get; }
        public ExtremumInfo[] RawExtrema { get; }
        public ExtremumInfo[] StageTwoExtrema { get; }
        public GraphPointInfo[] GraphPoints { get; }
        public DelayMarkerInfo[] DelayMarkers { get; }
        public double[] DelayProfile { get; }
        public int TemporalShiftFrames { get; }
        public int Skip { get; }
        public int VisibleCount => Math.Max(0, Frames.Length - Skip);
    }

    private sealed class VerticalProgressBar : Control
    {
        private int _minimum;
        private int _maximum = 100;
        private int _value;

        public VerticalProgressBar()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
            TabStop = false;
            ForeColor = SystemColors.Highlight;
            BackColor = SystemColors.ControlLight;
            MinimumSize = new Size(14, 80);
        }

        public int Minimum
        {
            get => _minimum;
            set
            {
                if (_minimum == value)
                {
                    return;
                }

                _minimum = value;
                if (_maximum < _minimum)
                {
                    _maximum = _minimum;
                }
                Value = _value;
                Invalidate();
            }
        }

        public int Maximum
        {
            get => _maximum;
            set
            {
                if (_maximum == value)
                {
                    return;
                }

                _maximum = value;
                if (_maximum < _minimum)
                {
                    _minimum = _maximum;
                }
                Value = _value;
                Invalidate();
            }
        }

        public int Value
        {
            get => _value;
            set
            {
                int clamped = Math.Min(_maximum, Math.Max(_minimum, value));
                if (_value == clamped)
                {
                    return;
                }

                _value = clamped;
                Invalidate();
            }
        }

        protected override Size DefaultSize => new(18, 140);

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            Rectangle rect = ClientRectangle;
            if (rect.Width <= 2 || rect.Height <= 2)
            {
                return;
            }

            using var background = new SolidBrush(BackColor);
            Rectangle inner = Rectangle.Inflate(rect, -1, -1);
            e.Graphics.FillRectangle(background, inner);
            ControlPaint.DrawBorder(e.Graphics, rect, SystemColors.ControlDark, ButtonBorderStyle.Solid);

            int range = _maximum - _minimum;
            if (range <= 0)
            {
                return;
            }

            double percent = (_value - _minimum) / (double)range;
            int fillHeight = (int)Math.Round(inner.Height * percent);
            if (fillHeight <= 0)
            {
                return;
            }

            Rectangle fillRect = new(inner.Left, inner.Bottom - fillHeight, inner.Width, fillHeight);
            using var fill = new SolidBrush(ForeColor);
            e.Graphics.FillRectangle(fill, fillRect);
        }
    }

    private sealed class PredictionCanvas : Control
    {
        public const int WindowSize = 300;
        private const int Stage1RangeWarmup = 20;

        private readonly PlotModel _model;
        private int _windowStart;
        private double[] _windowFrames = Array.Empty<double>();
        private double[] _windowProcessedValue = Array.Empty<double>();
        private Rectangle _stage3Rect;
        private int _lastHoverFrame = -1;
        private double? _lastHoverValue;

        public PredictionCanvas(PlotModel model)
        {
            _model = model;
            DoubleBuffered = true;
            ResizeRedraw = true;
        }

        public event EventHandler<FrameEventArgs>? HoveredFrameChanged;

        public void SetWindowStart(int start)
        {
            int visible = _model.VisibleCount;
            if (visible <= 0)
            {
                _windowStart = 0;
            }
            else
            {
                int maxStart = Math.Max(0, visible - Math.Min(WindowSize, visible));
                _windowStart = Math.Max(0, Math.Min(start, maxStart));
            }
            _windowFrames = Array.Empty<double>();
            _windowProcessedValue = Array.Empty<double>();
            _stage3Rect = Rectangle.Empty;
            Invalidate();
            NotifyHoverFrame(-1, null);
        }


        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (_windowFrames.Length == 0 || _stage3Rect.Width <= 0)
            {
                NotifyHoverFrame(-1, null);
                return;
            }

            if (!_stage3Rect.Contains(e.Location))
            {
                NotifyHoverFrame(-1, null);
                return;
            }

            double minX = _windowFrames[0];
            double maxX = _windowFrames[^1];
            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                NotifyHoverFrame(-1, null);
                return;
            }

            double ratio = (e.X - _stage3Rect.Left) / (double)_stage3Rect.Width;
            ratio = Math.Clamp(ratio, 0.0, 1.0);
            int frame = (int)Math.Round(minX + ratio * spanX);
            double? processedValue = null;
            if (_windowProcessedValue.Length > 0)
            {
                int valueIndex = (int)Math.Round(ratio * (_windowProcessedValue.Length - 1));
                valueIndex = Math.Clamp(valueIndex, 0, _windowProcessedValue.Length - 1);
                processedValue = _windowProcessedValue[valueIndex];
            }
            NotifyHoverFrame(frame, processedValue);
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            base.OnMouseLeave(e);
            NotifyHoverFrame(-1, null);
        }

        private void NotifyHoverFrame(int frameIndex, double? processedValue)
        {
            if (frameIndex == _lastHoverFrame && System.Nullable.Equals(_lastHoverValue, processedValue))
            {
                return;
            }

            _lastHoverFrame = frameIndex;
            _lastHoverValue = processedValue;
            HoveredFrameChanged?.Invoke(this, new FrameEventArgs(frameIndex, processedValue));
        }

        public (int Start, int End) GetWindowFrameRange()
        {
            int visible = _model.VisibleCount;
            if (visible <= 0)
            {
                return (0, -1);
            }

            int windowLength = Math.Min(WindowSize, visible);
            int startIndex = Math.Max(0, Math.Min(_windowStart, visible - windowLength));
            int endIndex = startIndex + windowLength - 1;

            int frameStartIndex = Math.Min(_model.Frames.Length - 1, _model.Skip + startIndex);
            int frameEndIndex = Math.Min(_model.Frames.Length - 1, _model.Skip + endIndex);
            int globalStart = (int)Math.Round(_model.Frames[frameStartIndex]);
            int globalEnd = (int)Math.Round(_model.Frames[frameEndIndex]);
            return (globalStart, globalEnd);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            int visible = _model.VisibleCount;
            if (visible <= 1)
            {
                _windowFrames = Array.Empty<double>();
                _windowProcessedValue = Array.Empty<double>();
                _stage3Rect = Rectangle.Empty;
                return;
            }

            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;

            Rectangle bounds = ClientRectangle;
            Rectangle plotArea = Rectangle.Inflate(bounds, -60, -52);
            if (plotArea.Width <= 0 || plotArea.Height <= 0)
            {
                return;
            }

            const int sections = 3;
            const int gap = 18;
            int availableHeight = plotArea.Height - gap * (sections - 1);
            if (availableHeight <= sections)
            {
                return;
            }

            int baseHeight = availableHeight / sections;
            int remainder = availableHeight % sections;
            var stageRects = new Rectangle[sections];
            int currentTop = plotArea.Top;
            for (int i = 0; i < sections; i++)
            {
                int height = baseHeight + (i < remainder ? 1 : 0);
                stageRects[i] = new Rectangle(plotArea.Left, currentTop, plotArea.Width, height);
                currentTop += height + gap;
            }

            int skip = _model.Skip;
            double[] frames = Slice(_model.Frames, skip, visible);
            double[] rawChange = Slice(_model.RawChange, skip, visible);
            double[] rawSignal = Slice(_model.RawSignal, skip, visible);
            double[] denoisedSignal = Slice(_model.Denoised, skip, visible);
            double[] smoothed = Slice(_model.Smoothed, skip, visible);
            double[] processedValue = Slice(_model.ProcessedValue, skip, visible);
            double[] phaseMarker = Slice(_model.PhaseMarker, skip, visible);

            int windowLength = Math.Min(WindowSize, visible);
            int windowStart = Math.Max(0, Math.Min(_windowStart, visible - windowLength));
            double[] windowFrames = Slice(frames, windowStart, windowLength);
            double[] windowRawChange = Slice(rawChange, windowStart, windowLength);
            double[] windowDenoised = Slice(denoisedSignal, windowStart, windowLength);
            double[] windowProcessedValue = Slice(processedValue, windowStart, windowLength);
            double[] windowPhaseMarker = Slice(phaseMarker, windowStart, windowLength);

            _windowFrames = windowFrames;
            _windowProcessedValue = windowProcessedValue;

            _windowFrames = windowFrames;
            _stage3Rect = stageRects[2];

            var rawExtrema = _model.RawExtrema;
            var stageExtrema = _model.StageTwoExtrema;
            var graphPoints = _model.GraphPoints;
            var delayMarkers = _model.DelayMarkers;

            double windowMinX = windowFrames.Length > 0 ? windowFrames[0] : frames[0];
            double windowMaxX = windowFrames.Length > 0 ? windowFrames[^1] : frames[^1];

            var rawWindowExtrema = rawExtrema.Where(ext => ext.Frame >= windowMinX && ext.Frame <= windowMaxX).ToArray();
            var refinedWindowExtrema = stageExtrema.Where(ext => ext.Frame >= windowMinX && ext.Frame <= windowMaxX).ToArray();
            var windowGraphPoints = graphPoints.Where(pt => pt.Position >= windowMinX && pt.Position <= windowMaxX).ToArray();
            var windowDelayMarkers = delayMarkers.Where(m => m.Frame >= windowMinX && m.Frame <= windowMaxX).ToArray();

            var baseFont = Font ?? SystemFonts.MessageBoxFont;
            float titleSize = Math.Max(6f, baseFont.Size + 2f);
            float axisSize = Math.Max(6f, baseFont.Size - 1f);
            using var titleFont = new Font(baseFont.FontFamily, titleSize, FontStyle.Bold);
            using var axisFont = new Font(baseFont.FontFamily, axisSize, FontStyle.Regular);

            DrawStage1(g, stageRects[0], frames, rawSignal, denoisedSignal, smoothed, windowMinX, windowMaxX, titleFont, axisFont);
            DrawStage2(g, stageRects[1], windowFrames, windowRawChange, windowDenoised, rawWindowExtrema, refinedWindowExtrema, titleFont, axisFont);
            DrawStage3(g, stageRects[2], windowFrames, windowProcessedValue, windowPhaseMarker, refinedWindowExtrema, windowGraphPoints, windowDelayMarkers, titleFont, axisFont);
        }

        private static void DrawStage1(Graphics g, Rectangle rect, double[] frames, double[] raw, double[] denoised, double[] smoothed, double windowStart, double windowEnd, Font titleFont, Font axisFont)
        {
            if (frames.Length == 0)
            {
                return;
            }

            double[] trimmedRaw = TrimLeading(raw, Stage1RangeWarmup);
            double[] trimmedSmooth = TrimLeading(smoothed, Stage1RangeWarmup);
            double[] trimmedDenoised = TrimLeading(denoised, Stage1RangeWarmup);
            (double minY, double maxY) = ComputeRange(trimmedRaw, trimmedSmooth, trimmedDenoised);
            AddPadding(ref minY, ref maxY);
            double minX = frames[0];
            double maxX = frames[^1];
            if (Math.Abs(maxX - minX) < 1e-6)
            {
                maxX = minX + 1.0;
            }

            DrawPlotBackground(g, rect, "Stage 1 - Raw change", minX, maxX, minY, maxY, showXAxis: false, titleFont, axisFont);
            double overlayStart = Math.Min(windowStart, windowEnd);
            double overlayEnd = Math.Max(windowStart, windowEnd);
            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }

            double clampedStart = Math.Max(minX, overlayStart);
            double clampedEnd = Math.Min(maxX, overlayEnd);
            if (clampedEnd > clampedStart)
            {
                float left = (float)((clampedStart - minX) / spanX * rect.Width + rect.Left);
                float right = (float)((clampedEnd - minX) / spanX * rect.Width + rect.Left);
                float width = Math.Max(1f, right - left);
                using var overlayBrush = new SolidBrush(Color.FromArgb(48, 70, 130, 180));
                g.FillRectangle(overlayBrush, left, rect.Top, width, rect.Height);
                using var overlayPen = new Pen(Color.FromArgb(160, 70, 130, 180), 1f);
                g.DrawRectangle(overlayPen, left, rect.Top, width, rect.Height);
            }

            DrawLineSeries(g, rect, frames, raw, minX, maxX, minY, maxY, Color.SteelBlue, 1.6f);
            var legendItems = new List<(string Label, Color Color)>
            {
                ("Raw change", Color.SteelBlue)
            };
            if (denoised.Length > 0)
            {
                DrawLineSeries(g, rect, frames, denoised, minX, maxX, minY, maxY, Color.DarkOrange, 1.2f, DashStyle.Dash);
                legendItems.Add(("FFT denoised", Color.DarkOrange));
            }
            DrawLineSeries(g, rect, frames, smoothed, minX, maxX, minY, maxY, Color.MediumPurple, 1.2f);
            legendItems.Add(("Smoothed", Color.MediumPurple));
            DrawLegend(g, rect, axisFont, legendItems.ToArray());
        }

        private static void DrawStage2(Graphics g, Rectangle rect, double[] frames, double[] raw, double[] denoised, ExtremumInfo[] rawExtrema, ExtremumInfo[] refinedExtrema, Font titleFont, Font axisFont)
        {
            if (frames.Length == 0)
            {
                return;
            }

            (double minY, double maxY) = ComputeRange(raw, denoised);
            AddPadding(ref minY, ref maxY);
            double minX = frames[0];
            double maxX = frames[^1];
            if (Math.Abs(maxX - minX) < 1e-6)
            {
                maxX = minX + 1.0;
            }

            DrawPlotBackground(g, rect, "Stage 2 - Extrema refinement", minX, maxX, minY, maxY, showXAxis: false, titleFont, axisFont);
            DrawLineSeries(g, rect, frames, raw, minX, maxX, minY, maxY, Color.SteelBlue, 1.2f);
            if (denoised.Length > 0)
            {
                DrawLineSeries(g, rect, frames, denoised, minX, maxX, minY, maxY, Color.DarkGoldenrod, 1.1f, DashStyle.Dash);
            }
            DrawExtremaMarkers(g, rect, rawExtrema, minX, maxX, minY, maxY, Color.Firebrick, Color.DarkSlateBlue, 6f);
            DrawExtremaMarkers(g, rect, refinedExtrema, minX, maxX, minY, maxY, Color.DarkOrange, Color.Teal, 8f);
            var legendItems = new List<(string Label, Color Color)>
            {
                ("Raw change", Color.SteelBlue)
            };
            if (denoised.Length > 0)
            {
                legendItems.Add(("FFT denoised", Color.DarkGoldenrod));
            }
            legendItems.Add(("Raw peak", Color.Firebrick));
            legendItems.Add(("Raw trough", Color.DarkSlateBlue));
            legendItems.Add(("Refined peak", Color.DarkOrange));
            legendItems.Add(("Refined trough", Color.Teal));
            DrawLegend(g, rect, axisFont, legendItems.ToArray());
        }

        private static void DrawStage3(Graphics g, Rectangle rect, double[] frames, double[] processedValue, double[] phaseMarker, ExtremumInfo[] stageExtrema, GraphPointInfo[] graphPoints, DelayMarkerInfo[] delayMarkers, Font titleFont, Font axisFont)
        {
            if (frames.Length == 0)
            {
                return;
            }

            const double minY = 0.0;
            const double maxY = 100.0;
            double minX = frames[0];
            double maxX = frames[^1];
            if (Math.Abs(maxX - minX) < 1e-6)
            {
                maxX = minX + 1.0;
            }

            DrawPlotBackground(g, rect, "Stage 3 - Graph points", minX, maxX, minY, maxY, showXAxis: true, titleFont, axisFont);
            DrawLineSeries(g, rect, frames, processedValue, minX, maxX, minY, maxY, Color.ForestGreen, 1.6f);
            DrawStageExtremaOverlay(g, rect, frames, processedValue, stageExtrema, minX, maxX, minY, maxY);
            DrawGraphPointMarkers(g, rect, graphPoints, minX, maxX, minY, maxY);
            DrawDelayMarkers(g, rect, delayMarkers, minX, maxX, axisFont);
            DrawPhaseMarkers(g, rect, frames, phaseMarker, minX, maxX);
            DrawLegend(g, rect, axisFont,
                ("Processed value", Color.ForestGreen),
                ("Original peak", Color.DarkOrange),
                ("Original trough", Color.Teal),
                ("Peak", Color.Firebrick),
                ("Trough", Color.MidnightBlue),
                ("Intermediate", Color.Gray),
                ("Boosted", Color.MediumPurple),
                ("Delay marker", Color.DarkSlateGray),
                ("Central adjusted", Color.Goldenrod));
        }

        private static void DrawPlotBackground(Graphics g, Rectangle rect, string title, double minX, double maxX, double minY, double maxY, bool showXAxis, Font titleFont, Font axisFont)
        {
            g.FillRectangle(Brushes.White, rect);
            using var borderPen = new Pen(Color.Gainsboro, 1f);
            g.DrawRectangle(borderPen, rect);

            var titlePosition = new PointF(rect.Left, rect.Top - titleFont.Height - 4);
            g.DrawString(title, titleFont, Brushes.Black, titlePosition);

            double ySpan = maxY - minY;
            if (ySpan <= 0)
            {
                ySpan = 1.0;
            }

            string maxLabel = maxY.ToString("0.###");
            string minLabel = minY.ToString("0.###");
            var maxSize = g.MeasureString(maxLabel, axisFont);
            g.DrawString(maxLabel, axisFont, Brushes.DimGray, rect.Left - maxSize.Width - 6, rect.Top - axisFont.Height / 2f);
            var minSize = g.MeasureString(minLabel, axisFont);
            g.DrawString(minLabel, axisFont, Brushes.DimGray, rect.Left - minSize.Width - 6, rect.Bottom - minSize.Height);

            if (showXAxis)
            {
                string minXLabel = minX.ToString("0");
                string maxXLabel = maxX.ToString("0");
                g.DrawString(minXLabel, axisFont, Brushes.DimGray, rect.Left, rect.Bottom + 4);
                var maxXSize = g.MeasureString(maxXLabel, axisFont);
                g.DrawString(maxXLabel, axisFont, Brushes.DimGray, rect.Right - maxXSize.Width, rect.Bottom + 4);
                var captionSize = g.MeasureString("Frame", axisFont);
                g.DrawString("Frame", axisFont, Brushes.DimGray, rect.Left + (rect.Width - captionSize.Width) / 2f, rect.Bottom + 4);
            }

            if (minY < 0 && maxY > 0)
            {
                double ratio = (0 - minY) / (maxY - minY);
                float zeroY = (float)(rect.Bottom - ratio * rect.Height);
                using var zeroPen = new Pen(Color.LightGray, 1f) { DashStyle = DashStyle.Dot };
                g.DrawLine(zeroPen, rect.Left, zeroY, rect.Right, zeroY);
            }
        }

        private static void DrawLineSeries(Graphics g, Rectangle rect, double[] x, double[] y, double minX, double maxX, double minY, double maxY, Color color, float width, DashStyle? dashStyle = null)
        {
            int count = Math.Min(x.Length, y.Length);
            if (count == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }
            double spanY = maxY - minY;
            if (spanY <= 0)
            {
                spanY = 1.0;
            }

            using var pen = new Pen(color, width);
            if (dashStyle.HasValue)
            {
                pen.DashStyle = dashStyle.Value;
            }
            PointF? previous = null;
            for (int i = 0; i < count; i++)
            {
                double xv = x[i];
                double yv = y[i];
                if (double.IsNaN(xv) || double.IsNaN(yv) || double.IsInfinity(xv) || double.IsInfinity(yv))
                {
                    previous = null;
                    continue;
                }

                float px = (float)((xv - minX) / spanX * rect.Width + rect.Left);
                float py = (float)(rect.Bottom - (yv - minY) / spanY * rect.Height);
                if (float.IsNaN(px) || float.IsNaN(py) || float.IsInfinity(px) || float.IsInfinity(py))
                {
                    previous = null;
                    continue;
                }

                bool inside = px >= rect.Left && px <= rect.Right && py >= rect.Top && py <= rect.Bottom;
                if (!inside)
                {
                    previous = null;
                    continue;
                }

                var current = new PointF(px, py);
                if (previous.HasValue)
                {
                    g.DrawLine(pen, previous.Value, current);
                }
                previous = current;
            }
        }

        private static void DrawExtremaMarkers(Graphics g, Rectangle rect, ExtremumInfo[] extrema, double minX, double maxX, double minY, double maxY, Color peakColor, Color troughColor, float diameter)
        {
            if (extrema.Length == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }
            double spanY = maxY - minY;
            if (spanY <= 0)
            {
                spanY = 1.0;
            }

            using var peakBrush = new SolidBrush(Color.FromArgb(200, peakColor));
            using var troughBrush = new SolidBrush(Color.FromArgb(200, troughColor));
            using var peakPen = new Pen(peakColor, 1.0f);
            using var troughPen = new Pen(troughColor, 1.0f);

            foreach (var extremum in extrema)
            {
                double xVal = extremum.Frame;
                double yVal = extremum.RawChange;
                if (double.IsNaN(xVal) || double.IsNaN(yVal) || double.IsInfinity(xVal) || double.IsInfinity(yVal))
                {
                    continue;
                }
                if (xVal < minX || xVal > maxX)
                {
                    continue;
                }

                float px = (float)((xVal - minX) / spanX * rect.Width + rect.Left);
                float py = (float)(rect.Bottom - (yVal - minY) / spanY * rect.Height);
                if (float.IsNaN(px) || float.IsNaN(py) || float.IsInfinity(px) || float.IsInfinity(py))
                {
                    continue;
                }

                var markerRect = new RectangleF(px - diameter / 2f, py - diameter / 2f, diameter, diameter);
                if (string.Equals(extremum.Kind, "peak", StringComparison.OrdinalIgnoreCase))
                {
                    g.FillEllipse(peakBrush, markerRect);
                    g.DrawEllipse(peakPen, markerRect);
                }
                else
                {
                    g.FillEllipse(troughBrush, markerRect);
                    g.DrawEllipse(troughPen, markerRect);
                }
            }
        }

        private static void DrawGraphPointMarkers(Graphics g, Rectangle rect, GraphPointInfo[] points, double minX, double maxX, double minY, double maxY)
        {
            if (points.Length == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }
            double spanY = maxY - minY;
            if (spanY <= 0)
            {
                spanY = 1.0;
            }

            using var peakBrush = new SolidBrush(Color.Firebrick);
            using var troughBrush = new SolidBrush(Color.MidnightBlue);
            using var intermediateBrush = new SolidBrush(Color.Gray);
            using var boostedBrush = new SolidBrush(Color.MediumPurple);
            using var adjustedBrush = new SolidBrush(Color.Goldenrod);
            using var borderPen = new Pen(Color.White, 1f);

            foreach (var point in points)
            {
                double x = point.Position;
                if (x < minX || x > maxX)
                {
                    continue;
                }

                double y = point.Value;
                if (double.IsNaN(y) || double.IsInfinity(y))
                {
                    continue;
                }

                float px = (float)((x - minX) / spanX * rect.Width + rect.Left);
                float py = (float)(rect.Bottom - (y - minY) / spanY * rect.Height);

                bool isAdjusted = !string.IsNullOrEmpty(point.Origin) && point.Origin.IndexOf("central_adjusted", StringComparison.OrdinalIgnoreCase) >= 0;

                if (isAdjusted)
                {
                    float size = 10f;
                    var rectMarker = new RectangleF(px - size / 2f, py - size / 2f, size, size);
                    g.FillRectangle(adjustedBrush, rectMarker);
                    using var outlinePen = new Pen(Color.DarkGoldenrod, 1.2f);
                    g.DrawRectangle(outlinePen, rectMarker.X, rectMarker.Y, rectMarker.Width, rectMarker.Height);
                    continue;
                }

                int diameter = point.Label switch
                {
                    "boosted" => 8,
                    "intermediate" => 6,
                    _ => 9,
                };

                Brush brush = point.Label switch
                {
                    "peak" => peakBrush,
                    "trough" => troughBrush,
                    "boosted" => boostedBrush,
                    _ => intermediateBrush,
                };

                var markerRect = new RectangleF(px - diameter / 2f, py - diameter / 2f, diameter, diameter);
                g.FillEllipse(brush, markerRect);
                g.DrawEllipse(borderPen, markerRect);
            }
        }

        private static void DrawStageExtremaOverlay(Graphics g, Rectangle rect, double[] frames, double[] values, ExtremumInfo[] extrema, double minX, double maxX, double minY, double maxY)
        {
            if (extrema.Length == 0 || frames.Length == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }
            double spanY = maxY - minY;
            if (spanY <= 0)
            {
                spanY = 1.0;
            }

            using var peakPen = new Pen(Color.DarkOrange, 1.6f);
            using var troughPen = new Pen(Color.Teal, 1.6f);
            using var peakFill = new SolidBrush(Color.FromArgb(60, Color.DarkOrange));
            using var troughFill = new SolidBrush(Color.FromArgb(60, Color.Teal));

            foreach (var ext in extrema)
            {
                double x = ext.Frame;
                if (x < minX || x > maxX)
                {
                    continue;
                }

                double y = InterpolateSeries(frames, values, x);
                if (double.IsNaN(y) || double.IsInfinity(y))
                {
                    continue;
                }

                float px = (float)((x - minX) / spanX * rect.Width + rect.Left);
                float py = (float)(rect.Bottom - (y - minY) / spanY * rect.Height);

                float size = 10f;
                var markerRect = new RectangleF(px - size / 2f, py - size / 2f, size, size);
                bool isPeak = string.Equals(ext.Kind, "peak", StringComparison.OrdinalIgnoreCase);
                if (isPeak)
                {
                    g.FillEllipse(peakFill, markerRect);
                    g.DrawEllipse(peakPen, markerRect);
                }
                else
                {
                    g.FillEllipse(troughFill, markerRect);
                    g.DrawEllipse(troughPen, markerRect);
                }
            }
        }

        private static double InterpolateSeries(double[] x, double[] y, double target)
        {
            if (x.Length == 0 || y.Length == 0)
            {
                return double.NaN;
            }

            if (target <= x[0])
            {
                return y[0];
            }

            if (target >= x[^1])
            {
                return y[^1];
            }

            int index = Array.BinarySearch(x, target);
            if (index >= 0)
            {
                if (index >= y.Length)
                {
                    return y[^1];
                }
                return y[index];
            }

            int next = ~index;
            int prev = Math.Max(0, next - 1);
            if (next >= x.Length)
            {
                return y[^1];
            }

            double x0 = x[prev];
            double x1 = x[next];
            double y0 = y[Math.Min(prev, y.Length - 1)];
            double y1 = y[Math.Min(next, y.Length - 1)];
            double span = x1 - x0;
            if (span <= 1e-9)
            {
                return y0;
            }
            double alpha = (target - x0) / span;
            return y0 + alpha * (y1 - y0);
        }


        private static void DrawDelayMarkers(Graphics g, Rectangle rect, DelayMarkerInfo[] markers, double minX, double maxX, Font font)
        {
            if (markers.Length == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }

            const float triangleHeight = 7f;
            const float triangleHalfWidth = 4f;
            float baseline = rect.Top + 18f;

            using var brush = new SolidBrush(Color.DarkSlateGray);
            using var pen = new Pen(Color.DarkSlateGray, 1f);

            float lastLabelRight = float.NegativeInfinity;

            foreach (var marker in markers.OrderBy(m => m.Frame))
            {
                double frame = marker.Frame;
                if (frame < minX || frame > maxX)
                {
                    continue;
                }

                float px = (float)((frame - minX) / spanX * rect.Width + rect.Left);
                if (float.IsNaN(px) || float.IsInfinity(px))
                {
                    continue;
                }

                var points = new[]
                {
                    new PointF(px, baseline),
                    new PointF(px - triangleHalfWidth, baseline - triangleHeight),
                    new PointF(px + triangleHalfWidth, baseline - triangleHeight)
                };
                g.FillPolygon(brush, points);
                g.DrawPolygon(pen, points);

                string label = marker.Delay.ToString("0.#");
                var size = g.MeasureString(label, font);
                float textX = px - size.Width / 2f;
                textX = Math.Max(rect.Left, Math.Min(textX, rect.Right - size.Width));
                if (textX <= lastLabelRight + 4f)
                {
                    continue;
                }

                float textY = baseline - triangleHeight - size.Height - 2f;
                if (textY < rect.Top)
                {
                    textY = rect.Top;
                }

                g.DrawString(label, font, Brushes.DimGray, textX, textY);
                lastLabelRight = textX + size.Width;
            }
        }

        private static void DrawPhaseMarkers(Graphics g, Rectangle rect, double[] frames, double[] markers, double minX, double maxX)
        {
            if (markers.Length == 0)
            {
                return;
            }

            double spanX = maxX - minX;
            if (spanX <= 0)
            {
                spanX = 1.0;
            }

            float top = rect.Top + 3f;
            float bottom = rect.Bottom - 3f;

            using var peakPen = new Pen(Color.Firebrick, 1.2f);
            using var troughPen = new Pen(Color.MidnightBlue, 1.2f);

            int count = Math.Min(frames.Length, markers.Length);
            for (int i = 0; i < count; i++)
            {
                double frame = frames[i];
                double marker = markers[i];
                if (double.IsNaN(frame) || double.IsNaN(marker) || double.IsInfinity(frame) || double.IsInfinity(marker))
                {
                    continue;
                }

                float px = (float)((frame - minX) / spanX * rect.Width + rect.Left);
                if (float.IsNaN(px) || float.IsInfinity(px))
                {
                    continue;
                }

                var pen = marker >= 50.0 ? peakPen : troughPen;
                if (px < rect.Left || px > rect.Right)
                {
                    continue;
                }
                g.DrawLine(pen, px, top, px, bottom);
            }
        }

        private static void DrawLegend(Graphics g, Rectangle rect, Font font, params (string Label, Color Color)[] entries)
        {
            if (entries.Length == 0)
            {
                return;
            }

            const int padding = 6;
            const int boxSize = 10;

            int totalWidth = 0;
            foreach (var entry in entries)
            {
                int labelWidth = (int)Math.Ceiling(g.MeasureString(entry.Label, font).Width);
                totalWidth += boxSize + 4 + labelWidth + padding;
            }

            int startX = Math.Max(rect.Left + padding, rect.Right - totalWidth);
            int y = rect.Top + padding;

            using var borderPen = new Pen(Color.Gray, 1f);
            foreach (var entry in entries)
            {
                int labelWidth = (int)Math.Ceiling(g.MeasureString(entry.Label, font).Width);
                using var brush = new SolidBrush(entry.Color);
                g.FillRectangle(brush, startX, y, boxSize, boxSize);
                g.DrawRectangle(borderPen, startX, y, boxSize, boxSize);
                g.DrawString(entry.Label, font, Brushes.Black, startX + boxSize + 4, y - 1);
                startX += boxSize + 4 + labelWidth + padding;
            }
        }

        private static double[] Slice(double[] source, int start, int count)
        {
            if (source.Length == 0 || count <= 0)
            {
                return Array.Empty<double>();
            }

            if (start < 0)
            {
                start = 0;
            }
            if (start >= source.Length)
            {
                return Array.Empty<double>();
            }

            int actual = Math.Min(count, source.Length - start);
            var result = new double[actual];
            Array.Copy(source, start, result, 0, actual);
            return result;
        }

        private static (double Min, double Max) ComputeRange(params double[][] series)
        {
            double min = double.PositiveInfinity;
            double max = double.NegativeInfinity;
            foreach (var arr in series)
            {
                foreach (double value in arr)
                {
                    if (double.IsNaN(value) || double.IsInfinity(value))
                    {
                        continue;
                    }
                    if (value < min)
                    {
                        min = value;
                    }
                    if (value > max)
                    {
                        max = value;
                    }
                }
            }

            if (double.IsInfinity(min) || double.IsInfinity(max))
            {
                return (-1.0, 1.0);
            }

            if (Math.Abs(max - min) < 1e-12)
            {
                double center = (max + min) / 2.0;
                double epsilon = Math.Max(Math.Abs(center) * 1e-12, 1e-9);
                return (center - epsilon, center + epsilon);
            }

            return (min, max);
        }

        private static void AddPadding(ref double min, ref double max)
        {
            double span = max - min;
            if (span <= 0)
            {
                double center = min;
                double epsilon = Math.Max(Math.Abs(center) * 1e-12, 1e-9);
                min = center - epsilon;
                max = center + epsilon;
            }
        }



        public readonly struct FrameEventArgs
        {
            public FrameEventArgs(int frameIndex, double? processedValue)
            {
                FrameIndex = frameIndex;
                ProcessedValue = processedValue;
            }

            public int FrameIndex { get; }
            public double? ProcessedValue { get; }
        }
        private static double[] TrimLeading(double[] source, int count)
        {
            if (source.Length <= count || count <= 0)
            {
                return source;
            }

            var trimmed = new double[source.Length - count];
            Array.Copy(source, count, trimmed, 0, trimmed.Length);
            return trimmed;
        }
    }
}





















