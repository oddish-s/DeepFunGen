namespace app;

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

public partial class MainForm : Form
{
    private static readonly string[] SupportedVideoExtensions = { ".mp4", ".mov", ".avi", ".mkv" };

    private readonly BindingList<VideoJob> _jobs = new();
    private readonly HashSet<string> _queuedPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly VideoProcessingService _service;

    public MainForm()
    {
        InitializeComponent();

        _service = new VideoProcessingService();
        _service.Log += OnServiceLog;
        _service.PredictionReady += OnPredictionReady;
        _service.ExecutionProviderChanged += OnExecutionProviderChanged;

        ConfigureGrid();
        PopulateModelChoices();
        RefreshModelStatusLabel();
    }

    private void ConfigureGrid()
    {
        queueGrid.AutoGenerateColumns = false;
        queueGrid.ReadOnly = true;
        queueGrid.AllowUserToAddRows = false;
        queueGrid.AllowUserToDeleteRows = false;
        queueGrid.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
        queueGrid.MultiSelect = true;
        queueGrid.SelectionChanged += queueGrid_SelectionChanged;
        queueGrid.AllowDrop = true;
        queueGrid.DragEnter += MainForm_DragEnter;
        queueGrid.DragDrop += MainForm_DragDrop;
        queueGrid.CellDoubleClick += queueGrid_CellDoubleClick;

        queueGrid.Columns.Clear();
        queueGrid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Video",
            DataPropertyName = nameof(VideoJob.VideoPath),
            AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill,
        });
        queueGrid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Status",
            DataPropertyName = nameof(VideoJob.Status),
            Width = 90,
        });
        queueGrid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Progress",
            DataPropertyName = nameof(VideoJob.ProgressText),
            Width = 90,
        });
        queueGrid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Message",
            DataPropertyName = nameof(VideoJob.Message),
            AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill,
        });

        queueGrid.DataSource = _jobs;
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        base.OnFormClosing(e);
        _service.Dispose();
    }

    private void PopulateModelChoices()
    {
        modelComboBox.Items.Clear();
        modelComboBox.Enabled = true;
        modelComboBox.DisplayMember = nameof(ModelChoice.DisplayName);
        modelComboBox.ValueMember = nameof(ModelChoice.Path);

        var candidates = DiscoverModelPaths()
            .Select(path => new ModelChoice(Path.GetFileName(path), path))
            .ToList();

        foreach (var choice in candidates)
        {
            modelComboBox.Items.Add(choice);
        }

        if (candidates.Count > 0)
        {
            modelComboBox.SelectedIndex = 0;
        }
        else
        {
            modelComboBox.Items.Add("No ONNX models found");
            modelComboBox.SelectedIndex = 0;
            modelComboBox.Enabled = false;
        }

        RefreshModelStatusLabel();
    }

    private static IEnumerable<string> DiscoverModelPaths()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var root in EnumerateModelRoots())
        {
            if (!Directory.Exists(root))
            {
                continue;
            }

            foreach (var file in Directory.EnumerateFiles(root, "*.onnx", SearchOption.AllDirectories))
            {
                var full = Path.GetFullPath(file);
                if (seen.Add(full))
                {
                    yield return full;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateModelRoots()
    {
        var baseDir = AppContext.BaseDirectory;
        yield return baseDir;
        yield return Path.Combine(baseDir, "models");
    }

    private void addVideosButton_Click(object sender, EventArgs e)
    {
        using var dialog = new OpenFileDialog
        {
            Filter = "Video Files (*.mp4;*.mov;*.avi;*.mkv)|*.mp4;*.mov;*.avi;*.mkv|All Files|*.*",
            Title = "Select Videos",
            Multiselect = true,
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        QueueVideoPaths(dialog.FileNames);
    }

    private void QueueVideoPaths(IEnumerable<string> paths)
    {
        foreach (var path in paths)
        {
            TryQueueVideo(path);
        }
    }

    private void TryQueueVideo(string inputPath)
    {
        if (string.IsNullOrWhiteSpace(inputPath))
        {
            return;
        }

        if (Directory.Exists(inputPath))
        {
            var files = Directory.EnumerateFiles(inputPath, "*", SearchOption.AllDirectories)
                .Where(IsVideoFile)
                .ToList();
            if (files.Count == 0)
            {
                AppendLog(null, $"No supported videos in directory: {inputPath}");
                return;
            }
            QueueVideoPaths(files);
            return;
        }

        if (!File.Exists(inputPath))
        {
            AppendLog(null, $"File not found: {inputPath}");
            return;
        }

        if (!IsVideoFile(inputPath))
        {
            AppendLog(null, $"Unsupported file type: {inputPath}");
            return;
        }

        if (_queuedPaths.Contains(inputPath))
        {
            AppendLog(null, $"Already queued: {inputPath}");
            return;
        }

        var job = new VideoJob(inputPath);
        _jobs.Add(job);
        _queuedPaths.Add(inputPath);
        _service.Enqueue(job);
        AppendLog(job, "Queued");
    }

    private static bool IsVideoFile(string path)
    {
        var ext = Path.GetExtension(path);
        return !string.IsNullOrEmpty(ext) && SupportedVideoExtensions.Contains(ext, StringComparer.OrdinalIgnoreCase);
    }

    private void removeSelectedButton_Click(object sender, EventArgs e)
    {
        if (queueGrid.SelectedRows.Count == 0)
        {
            return;
        }

        var toRemove = new List<VideoJob>();
        foreach (DataGridViewRow row in queueGrid.SelectedRows)
        {
            if (row.DataBoundItem is VideoJob job && job.Status == VideoJobStatus.Pending)
            {
                toRemove.Add(job);
            }
        }

        if (toRemove.Count == 0)
        {
            return;
        }

        foreach (var job in toRemove)
        {
            _jobs.Remove(job);
            _queuedPaths.Remove(job.VideoPath);
        }

        _service.ClearPending();
    }

    private void clearFinishedButton_Click(object sender, EventArgs e)
    {
        var finished = _jobs
            .Where(j => j.Status is VideoJobStatus.Completed or VideoJobStatus.Skipped or VideoJobStatus.Failed)
            .ToList();
        foreach (var job in finished)
        {
            _jobs.Remove(job);
            _queuedPaths.Remove(job.VideoPath);
        }
    }

    private void modelComboBox_SelectedIndexChanged(object? sender, EventArgs e)
    {
        if (modelComboBox.SelectedItem is not ModelChoice choice)
        {
            return;
        }

        try
        {
            _service.UpdateModelPath(choice.Path);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Model Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        RefreshModelStatusLabel();
    }

    private void queueGrid_SelectionChanged(object? sender, EventArgs e)
    {
        // Selection no longer impacts post-processing; retained for future UI hooks.
    }

    private void queueGrid_CellDoubleClick(object? sender, DataGridViewCellEventArgs e)
    {
        if (e.RowIndex < 0)
        {
            return;
        }

        if (queueGrid.Rows[e.RowIndex].DataBoundItem is VideoJob job)
        {
            if (job.LatestSummary is { } summary)
            {
                ShowPrediction(summary);
            }
            else
            {
                AppendLog(job, "No processed result available yet.");
            }
        }
    }

    private void postprocessButton_Click(object? sender, EventArgs e)
    {
        var dialog = new PostprocessDialog(_service.PostprocessOptions);
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _service.PostprocessOptions = dialog.Options;
            AppendLog(null, "Updated post-processing options.");
        }
    }

    private void OnExecutionProviderChanged(string provider)
    {
        RefreshModelStatusLabel();
    }

    private void RefreshModelStatusLabel()
    {
        string modelName = modelComboBox.SelectedItem is ModelChoice choice ? choice.DisplayName : (modelComboBox.Enabled ? "Select model" : "No model");
        string provider = _service.ExecutionProvider;
        modelPathLabel.Text = $"{provider}";
    }

    private void MainForm_DragEnter(object? sender, DragEventArgs e)
    {
        if (e.Data?.GetDataPresent(DataFormats.FileDrop) == true)
        {
            e.Effect = DragDropEffects.Copy;
        }
        else
        {
            e.Effect = DragDropEffects.None;
        }
    }

    private void MainForm_DragDrop(object? sender, DragEventArgs e)
    {
        if (e.Data?.GetData(DataFormats.FileDrop) is not string[] paths || paths.Length == 0)
        {
            return;
        }

        QueueVideoPaths(paths);
    }

    private void OnServiceLog(VideoJob? job, string message)
    {
        AppendLog(job, message);
    }

    private void OnPredictionReady(VideoProcessingService.PredictionSummary summary)
    {
        summary.Job.LatestSummary = summary;
        ShowPrediction(summary);
    }

    private void ShowPrediction(VideoProcessingService.PredictionSummary summary)
    {
        summary.Job.LatestSummary = summary;
        var viewer = new PredictionViewerForm(summary);
        viewer.Show(this);
    }

    private void AppendLog(VideoJob? job, string message)
    {
        var builder = new StringBuilder();
        builder.Append('[');
        builder.Append(DateTime.Now.ToString("HH:mm:ss"));
        builder.Append("] ");
        if (job != null && !string.IsNullOrEmpty(job.VideoPath))
        {
            builder.Append(Path.GetFileName(job.VideoPath));
            builder.Append(" :: ");
        }
        builder.Append(message);
        builder.AppendLine();
        logTextBox.AppendText(builder.ToString());
    }

    private sealed record ModelChoice(string DisplayName, string Path)
    {
        public override string ToString() => DisplayName;
    }
}
