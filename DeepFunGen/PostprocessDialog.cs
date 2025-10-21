namespace app;

using System;
using System.Windows.Forms;

public sealed class PostprocessDialog : Form
{
    private readonly NumericUpDown _smoothWindowNumeric;
    private readonly NumericUpDown _prominenceRatioNumeric;
    private readonly NumericUpDown _minProminenceNumeric;
    private readonly NumericUpDown _maxSlopeNumeric;
    private readonly NumericUpDown _boostSlopeNumeric;
    private readonly NumericUpDown _minSlopeNumeric;
    private readonly NumericUpDown _mergeThresholdNumeric;
    private readonly NumericUpDown _centralDeviationNumeric;
    private readonly CheckBox _fftDenoiseCheckBox;
    private readonly NumericUpDown _fftFramesNumeric;

    public PostprocessOptions Options { get; private set; }

    public PostprocessDialog(PostprocessOptions initialOptions)
    {
        Options = initialOptions ?? PostprocessOptions.Default;
        Text = "Post-processing Options";
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        MinimizeBox = false;
        StartPosition = FormStartPosition.CenterParent;
        AutoSize = true;
        AutoSizeMode = AutoSizeMode.GrowAndShrink;
        Padding = new Padding(10);

        var layout = new TableLayoutPanel
        {
            ColumnCount = 2,
            RowCount = 11,
            Dock = DockStyle.Fill,
            AutoSize = true,
        };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));

        _smoothWindowNumeric = CreateIntNumeric(1, 200, Options.SmoothWindowFrames);
        _prominenceRatioNumeric = CreateFloatNumeric(0.0m, 10.0m, (decimal)Options.ProminenceRatio, 3, 0.01m);
        _minProminenceNumeric = CreateFloatNumeric(0.0m, 100.0m, (decimal)Options.MinProminence, 2);
        _maxSlopeNumeric = CreateFloatNumeric(0.0m, 100.0m, (decimal)Options.MaxSlope, 2);
        _boostSlopeNumeric = CreateFloatNumeric(0.0m, 100.0m, (decimal)Options.BoostSlope, 2);
        _minSlopeNumeric = CreateFloatNumeric(0.0m, 100.0m, (decimal)Options.MinSlope, 2);
        _mergeThresholdNumeric = CreateFloatNumeric(0.0m, 1000.0m, (decimal)Options.MergeThresholdMs, 1, 5m);
        _centralDeviationNumeric = CreateFloatNumeric(0.0m, 1.0m, (decimal)Options.CentralDeviationThreshold, 3, 0.01m);
        _fftDenoiseCheckBox = new CheckBox
        {
            Checked = Options.FftDenoise,
            AutoSize = true,
        };
        _fftFramesNumeric = CreateIntNumeric(4, 400, Options.FftFramesPerComponent);

        AddRow(layout, "Smooth Window Frames", _smoothWindowNumeric);
        AddRow(layout, "Prominence Ratio", _prominenceRatioNumeric);
        AddRow(layout, "Min Prominence", _minProminenceNumeric);
        AddRow(layout, "Max Slope", _maxSlopeNumeric);
        AddRow(layout, "Boost Slope", _boostSlopeNumeric);
        AddRow(layout, "Min Slope", _minSlopeNumeric);
        AddRow(layout, "Merge Threshold (ms)", _mergeThresholdNumeric);
        AddRow(layout, "Central Deviation Threshold", _centralDeviationNumeric);
        AddRow(layout, "FFT Denoise", _fftDenoiseCheckBox);
        AddRow(layout, "FFT Frames / Component", _fftFramesNumeric);

        var buttonPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.RightToLeft,
            AutoSize = true,
            Padding = new Padding(0, 10, 0, 0),
        };

        var saveButton = new Button
        {
            Text = "Save",
            AutoSize = true,
            DialogResult = DialogResult.OK,
        };
        saveButton.Click += (_, _) => OnSave();

        buttonPanel.Controls.Add(saveButton);

        layout.Controls.Add(buttonPanel, 0, 10);
        layout.SetColumnSpan(buttonPanel, 2);

        Controls.Add(layout);

        AcceptButton = saveButton;
    }

    private static NumericUpDown CreateIntNumeric(int minimum, int maximum, int value)
    {
        return new NumericUpDown
        {
            Minimum = minimum,
            Maximum = maximum,
            Value = Math.Min(Math.Max(value, minimum), maximum),
            Width = 120,
        };
    }

    private static NumericUpDown CreateFloatNumeric(decimal minimum, decimal maximum, decimal value, int decimalPlaces, decimal increment = 0.1m)
    {
        return new NumericUpDown
        {
            Minimum = minimum,
            Maximum = maximum,
            DecimalPlaces = decimalPlaces,
            Increment = increment,
            Value = Math.Min(Math.Max(value, minimum), maximum),
            Width = 120,
        };
    }

    private static void AddRow(TableLayoutPanel layout, string labelText, Control control)
    {
        var label = new Label
        {
            Text = labelText,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Padding = new Padding(0, 6, 10, 0),
        };

        layout.Controls.Add(label);
        layout.Controls.Add(control);
    }

    private void OnSave()
    {
        var previous = Options ?? PostprocessOptions.Default;
        Options = new PostprocessOptions
        {
            SmoothWindowFrames = (int)_smoothWindowNumeric.Value,
            ProminenceRatio = (double)_prominenceRatioNumeric.Value,
            MinProminence = (double)_minProminenceNumeric.Value,
            MaxSlope = (double)_maxSlopeNumeric.Value,
            BoostSlope = (double)_boostSlopeNumeric.Value,
            MinSlope = (double)_minSlopeNumeric.Value,
            MergeThresholdMs = (double)_mergeThresholdNumeric.Value,
            CentralDeviationThreshold = (double)_centralDeviationNumeric.Value,
            FftDenoise = _fftDenoiseCheckBox.Checked,
            FftFramesPerComponent = (int)_fftFramesNumeric.Value,
            FftWindowFrames = previous.FftWindowFrames,
        };

        DialogResult = DialogResult.OK;
        Close();
    }
}



