namespace app;

using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

public class PostprocessOptions
{
    [JsonPropertyName("smooth_window_frames")]
    public int SmoothWindowFrames { get; set; } = 3;

    [JsonPropertyName("prominence_ratio")]
    public double ProminenceRatio { get; set; } = 0.001;

    [JsonPropertyName("min_prominence")]
    public double MinProminence { get; set; } = 0.0;

    [JsonPropertyName("max_slope")]
    public double MaxSlope { get; set; } = 10.0;

    [JsonPropertyName("boost_slope")]
    public double BoostSlope { get; set; } = 7.0;

    [JsonPropertyName("min_slope")]
    public double MinSlope { get; set; } = 2.0;

    [JsonPropertyName("merge_threshold_ms")]
    public double MergeThresholdMs { get; set; } = 120.0;

    [JsonPropertyName("central_deviation_threshold")]
    public double CentralDeviationThreshold { get; set; } = 0.03;


    public static PostprocessOptions Default => new();

    public static PostprocessOptions Load(string path)
    {
        if (!File.Exists(path))
        {
            return Default;
        }

        try
        {
            var json = File.ReadAllText(path);
            var options = JsonSerializer.Deserialize<PostprocessOptions>(json);
            return options ?? Default;
        }
        catch
        {
            return Default;
        }
    }

    public static void Save(string path, PostprocessOptions options)
    {
        var json = JsonSerializer.Serialize(options, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    public override string ToString() =>
        $"smooth={SmoothWindowFrames}, prominence_ratio={ProminenceRatio}, min_prominence={MinProminence}, max_slope={MaxSlope}, boost_slope={BoostSlope}, min_slope={MinSlope}, merge_threshold_ms={MergeThresholdMs}, central_deviation_threshold={CentralDeviationThreshold}";
}



