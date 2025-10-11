namespace app
{
    public class PeakProperties
    {
        public int Index;
        public double Height;
        public double Prominence;
        public int LeftBase;
        public int RightBase;

        // For widths:
        public double Width;
        public double WidthHeight;
        public double LeftIp;
        public double RightIp;
    }

    public static class FindPeaks
    {
        // ---------- Public API ----------

        /// <summary>
        /// Find peaks in 1D signal. Partial port of scipy.signal.find_peaks.
        /// Returns indices and dictionary-like properties packed into PeakProperties[].
        /// Supported selections: height (min,max), threshold (min,max), distance,
        /// prominence (min,max), width (min,max) via relHeight and wlen.
        /// plateauSize handling is minimal (detects plateau edges).
        /// </summary>
        public static (int[] peaks, PeakProperties[] properties) Find(
            double[] x,
            (double? min, double? max)? height = null,
            (double? min, double? max)? threshold = null,
            int? distance = null,
            (double? min, double? max)? prominence = null,
            (double? min, double? max)? width = null,
            int? wlen = null,
            double relHeight = 0.5,
            (int? min, int? max)? plateauSize = null
        )
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Length < 3) return (Array.Empty<int>(), Array.Empty<PeakProperties>());
            if (distance.HasValue && distance.Value < 1) throw new ArgumentException("distance must be >= 1");

            // 1) get local maxima and plateau edges
            var (peaks, leftEdges, rightEdges) = LocalMaxima1D(x);

            var propsList = CreateDefaultPropList(peaks, x);

            // plateau_size selection (minimal)
            if (plateauSize.HasValue)
            {
                int[] plateauSizes = rightEdges.Zip(leftEdges, (r, l) => r - l + 1).ToArray();
                var keep = SelectByInterval(plateauSizes.Select(v => (double)v).ToArray(), plateauSize.Value.min, plateauSize.Value.max);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keep);
            }

            // height selection
            if (height.HasValue)
            {
                double[] peakHeights = peaks.Select(i => x[i]).ToArray();
                var keep = SelectByInterval(peakHeights, height.Value.min, height.Value.max);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keep);
            }

            // threshold selection (vertical distance to neighbors)
            if (threshold.HasValue)
            {
                var (keepMask, leftThresh, rightThresh) = SelectByPeakThreshold(x, peaks, threshold.Value.min, threshold.Value.max);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keepMask);
            }

            // distance selection (greedy by height)
            if (distance.HasValue)
            {
                var keep = SelectByPeakDistance(peaks, peaks.Select(i => x[i]).ToArray(), distance.Value);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keep);
            }

            // prominences (needed by prominence or widths)
            double[] prominences = null;
            int[] leftBases = null, rightBases = null;
            if (prominence.HasValue || width.HasValue)
            {
                int wlenArg = ArgWlenAsExpected(wlen);
                (prominences, leftBases, rightBases) = PeakProminences(x, peaks, wlenArg);
                // attach to props
                for (int k = 0; k < peaks.Length; k++)
                {
                    propsList[k].Prominence = prominences[k];
                    propsList[k].LeftBase = leftBases[k];
                    propsList[k].RightBase = rightBases[k];
                }
            }

            // prominence selection
            if (prominence.HasValue)
            {
                double[] pvals = propsList.Select(p => p.Prominence).ToArray();
                var keep = SelectByInterval(pvals, prominence.Value.min, prominence.Value.max);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keep);
            }

            // width calculation & selection
            if (width.HasValue)
            {
                // ensure prominences & bases present
                if (prominences == null)
                {
                    int wlenArg = ArgWlenAsExpected(wlen);
                    (prominences, leftBases, rightBases) = PeakProminences(x, peaks, wlenArg);
                    for (int k = 0; k < peaks.Length; k++)
                    {
                        propsList[k].Prominence = prominences[k];
                        propsList[k].LeftBase = leftBases[k];
                        propsList[k].RightBase = rightBases[k];
                    }
                }

                var (widths, widthHeights, leftIps, rightIps) = PeakWidths(x, peaks, relHeight, (prominences, leftBases, rightBases));
                for (int k = 0; k < peaks.Length; k++)
                {
                    propsList[k].Width = widths[k];
                    propsList[k].WidthHeight = widthHeights[k];
                    propsList[k].LeftIp = leftIps[k];
                    propsList[k].RightIp = rightIps[k];
                }

                double[] wvals = propsList.Select(p => p.Width).ToArray();
                var keep = SelectByInterval(wvals, width.Value.min, width.Value.max);
                (peaks, propsList, leftEdges, rightEdges) = FilterByMask(peaks, propsList, leftEdges, rightEdges, keep);
            }

            // pack results
            return (peaks, propsList.ToArray());
        }

        /// <summary>
        /// Wrapper to call peak_prominences similarly to scipy.
        /// Returns prominences, left_bases, right_bases.
        /// </summary>
        public static (double[] prominences, int[] leftBases, int[] rightBases) PeakProminences(double[] x, int[] peaks, int wlen = -1)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (peaks == null) return (Array.Empty<double>(), Array.Empty<int>(), Array.Empty<int>());
            if (peaks.Length == 0) return (Array.Empty<double>(), Array.Empty<int>(), Array.Empty<int>());

            return _PeakProminencesCore(x, peaks, wlen);
        }

        /// <summary>
        /// Wrapper to call peak_widths similarly to scipy.
        /// Requires prominence_data tuple (prominences, leftBases, rightBases) if available.
        /// </summary>
        public static (double[] widths, double[] widthHeights, double[] leftIps, double[] rightIps)
            PeakWidths(double[] x, int[] peaks, double relHeight = 0.5, (double[] prominences, int[] leftBases, int[] rightBases)? prominenceData = null, int wlen = -1)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (peaks == null) return (Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>());
            if (peaks.Length == 0) return (Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>());

            (double[] prominences, int[] leftBases, int[] rightBases) = prominenceData ?? _PeakProminencesCore(x, peaks, wlen);
            return _PeakWidthsCore(x, peaks, relHeight, prominences, leftBases, rightBases);
        }

        // ---------- Internal helpers (port of scipy logic) ----------

        private static (int[] peaks, int[] leftEdges, int[] rightEdges) LocalMaxima1D(double[] x)
        {
            int n = x.Length;
            var peaks = new List<int>();
            var leftEdges = new List<int>();
            var rightEdges = new List<int>();

            int i = 1;
            while (i < n - 1)
            {
                if (x[i] > x[i - 1])
                {
                    // increasing; check for plateau
                    int j = i;
                    while (j + 1 < n && x[j + 1] == x[i]) j++;
                    if (x[j] > x[j + 1 < n ? j + 1 : j])
                    {
                        // plateau or strict peak. choose middle index
                        int left = i;
                        int right = j;
                        int mid = (left + right) / 2;
                        peaks.Add(mid);
                        leftEdges.Add(left);
                        rightEdges.Add(right);
                        i = j + 1;
                        continue;
                    }
                }
                i++;
            }

            return (peaks.ToArray(), leftEdges.ToArray(), rightEdges.ToArray());
        }

        // Create PeakProperties array from peaks and x
        private static List<PeakProperties> CreateDefaultPropList(int[] peaks, double[] x)
        {
            var list = new List<PeakProperties>(peaks.Length);
            foreach (var p in peaks)
            {
                list.Add(new PeakProperties
                {
                    Index = p,
                    Height = x[p],
                    Prominence = 0.0,
                    LeftBase = 0,
                    RightBase = x.Length - 1
                });
            }
            return list;
        }

        // Apply mask to arrays/lists
        private static (int[] peaks, List<PeakProperties> props, int[] leftEdges, int[] rightEdges)
            FilterByMask(int[] peaks, List<PeakProperties> props, int[] leftEdges, int[] rightEdges, bool[] keep)
        {
            var newPeaks = new List<int>();
            var newProps = new List<PeakProperties>();
            var newLeft = new List<int>();
            var newRight = new List<int>();
            for (int i = 0; i < keep.Length; i++)
            {
                if (keep[i])
                {
                    newPeaks.Add(peaks[i]);
                    newProps.Add(props[i]);
                    newLeft.Add(leftEdges[i]);
                    newRight.Add(rightEdges[i]);
                }
            }
            return (newPeaks.ToArray(), newProps, newLeft.ToArray(), newRight.ToArray());
        }

        // Select by [min,max] interval
        private static bool[] SelectByInterval(double[] values, double? min, double? max)
        {
            bool[] keep = new bool[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                bool ok = true;
                if (min.HasValue) ok &= (values[i] >= min.Value);
                if (max.HasValue) ok &= (values[i] <= max.Value);
                keep[i] = ok;
            }
            return keep;
        }

        // threshold selection
        // returns keep mask and left/right thresholds (distance to neighbors)
        private static (bool[] keep, double[] leftThresh, double[] rightThresh) SelectByPeakThreshold(double[] x, int[] peaks, double? tmin, double? tmax)
        {
            int m = peaks.Length;
            double[] left = new double[m];
            double[] right = new double[m];
            bool[] keep = Enumerable.Repeat(true, m).ToArray();

            for (int i = 0; i < m; i++)
            {
                int idx = peaks[i];
                double lt = x[idx] - x[idx - 1];
                double rt = x[idx] - x[idx + 1];
                left[i] = lt;
                right[i] = rt;
            }

            if (tmin.HasValue)
            {
                for (int i = 0; i < m; i++)
                {
                    double mn = Math.Min(left[i], right[i]);
                    if (mn < tmin.Value) keep[i] = false;
                }
            }
            if (tmax.HasValue)
            {
                for (int i = 0; i < m; i++)
                {
                    double mx = Math.Max(left[i], right[i]);
                    if (mx > tmax.Value) keep[i] = false;
                }
            }

            return (keep, left, right);
        }

        // distance selection (greedy by height descending)
        private static bool[] SelectByPeakDistance(int[] peaks, double[] peakHeights, int distance)
        {
            int n = peaks.Length;
            bool[] selected = new bool[n];
            if (n == 0) return selected;

            // order by height desc
            var idxs = Enumerable.Range(0, n).OrderByDescending(i => peakHeights[i]).ToArray();
            bool[] blocked = new bool[n]; // block by original positions? we'll block by index distance using peaks positions
            for (int k = 0; k < idxs.Length; k++)
            {
                int i = idxs[k];
                if (blocked[i]) continue;
                selected[i] = true;
                // block any peaks within distance in sample space
                for (int j = 0; j < n; j++)
                {
                    if (!blocked[j])
                    {
                        if (Math.Abs(peaks[i] - peaks[j]) < distance)
                            blocked[j] = true;
                    }
                }
            }
            return selected;
        }

        // Ensure wlen: -1 means None, else integer > 1
        private static int ArgWlenAsExpected(int? wlen)
        {
            if (!wlen.HasValue) return -1;
            if (wlen.Value <= 1) throw new ArgumentException("wlen must be larger than 1");
            return wlen.Value;
        }

        // Core prominence algorithm (approximation faithful to scipy logic)
        private static (double[] prominences, int[] leftBases, int[] rightBases) _PeakProminencesCore(double[] x, int[] peaks, int wlen)
        {
            int n = x.Length;
            int m = peaks.Length;
            double[] prominences = new double[m];
            int[] leftBases = new int[m];
            int[] rightBases = new int[m];

            // precompute window if wlen provided (centered on peak)
            for (int i = 0; i < m; i++)
            {
                int peak = peaks[i];
                int leftLimit = 0;
                int rightLimit = n - 1;
                if (wlen > 1)
                {
                    int half = (wlen - 1) / 2;
                    leftLimit = Math.Max(0, peak - half);
                    rightLimit = Math.Min(n - 1, peak + half);
                }

                // Search left: move left until you find a point higher than current peak OR you reach boundary
                double curPeak = x[peak];
                int left = peak;
                double leftMin = double.PositiveInfinity;
                // move leftwards, tracking minimum
                int idx = peak;
                while (idx > leftLimit)
                {
                    idx--;
                    leftMin = Math.Min(leftMin, x[idx]);
                    if (x[idx] > curPeak)
                    {
                        // found higher; stop and base is the minimum on the path to that higher peak, then base candidate is index of that min (closest to peak)
                        break;
                    }
                }
                // If we found a higher peak at idx and idx<peak then left base must be set to the highest minimal point between them:
                int leftBaseCandidate = leftLimit;
                if (idx >= leftLimit && x[idx] > curPeak)
                {
                    // find minimum between idx+1..peak inclusive and choose closest min to peak
                    double minVal = double.PositiveInfinity;
                    int minIdx = idx + 1;
                    for (int j = idx + 1; j <= peak; j++)
                    {
                        if (x[j] < minVal) { minVal = x[j]; minIdx = j; }
                    }
                    leftBaseCandidate = minIdx;
                }
                else
                {
                    // no higher found; choose index of minimum in leftLimit..peak
                    double minVal = double.PositiveInfinity; int minIdx = leftLimit;
                    for (int j = leftLimit; j <= peak; j++)
                    {
                        if (x[j] < minVal) { minVal = x[j]; minIdx = j; }
                    }
                    leftBaseCandidate = minIdx;
                }

                // Search right
                idx = peak;
                double rightMin = double.PositiveInfinity;
                while (idx < rightLimit)
                {
                    idx++;
                    rightMin = Math.Min(rightMin, x[idx]);
                    if (x[idx] > curPeak) break;
                }

                int rightBaseCandidate = rightLimit;
                if (idx <= rightLimit && x[idx] > curPeak)
                {
                    double minVal = double.PositiveInfinity; int minIdx = idx - 1;
                    for (int j = peak; j <= idx - 1; j++)
                    {
                        if (x[j] < minVal) { minVal = x[j]; minIdx = j; }
                    }
                    rightBaseCandidate = minIdx;
                }
                else
                {
                    double minVal = double.PositiveInfinity; int minIdx = peak;
                    for (int j = peak; j <= rightLimit; j++)
                    {
                        if (x[j] < minVal) { minVal = x[j]; minIdx = j; }
                    }
                    rightBaseCandidate = minIdx;
                }

                // pick higher base (the contour line)
                double leftBaseHeight = x[leftBaseCandidate];
                double rightBaseHeight = x[rightBaseCandidate];
                double contour = Math.Max(leftBaseHeight, rightBaseHeight);
                double prom = curPeak - contour;
                if (prom < 0) prom = 0; // guard

                prominences[i] = prom;
                leftBases[i] = leftBaseCandidate;
                rightBases[i] = rightBaseCandidate;
            }

            return (prominences, leftBases, rightBases);
        }

        // Core widths algorithm (interpolate at eval height)
        private static (double[] widths, double[] widthHeights, double[] leftIps, double[] rightIps)
            _PeakWidthsCore(double[] x, int[] peaks, double relHeight, double[] prominences, int[] leftBases, int[] rightBases)
        {
            int m = peaks.Length;
            double[] widths = new double[m];
            double[] widthHeights = new double[m];
            double[] leftIps = new double[m];
            double[] rightIps = new double[m];

            for (int i = 0; i < m; i++)
            {
                int peak = peaks[i];
                double hPeak = x[peak];
                double prom = prominences[i];
                double hEval = hPeak - prom * relHeight;
                widthHeights[i] = hEval;

                // left intersection: go left from peak until value < hEval or reach left base
                int leftBound = leftBases[i];
                int li = peak;
                while (li > leftBound && x[li] > hEval) li--;
                // linear interpolate between li and li+1 to get precise ip
                double leftIp;
                if (li == peak && x[li] <= hEval) // immediate case
                {
                    leftIp = li;
                }
                else
                {
                    // li is index where x[li] <= hEval OR li == leftBound
                    int i1 = Math.Max(li, leftBound);
                    int i2 = Math.Min(li + 1, peak);
                    if (i2 == i1) leftIp = i1;
                    else
                    {
                        double y1 = x[i1], y2 = x[i2];
                        // y1 <= hEval <= y2
                        double denom = (y2 - y1);
                        double frac = denom == 0 ? 0.0 : (hEval - y1) / denom;
                        leftIp = i1 + frac;
                    }
                }

                // right intersection
                int rightBound = rightBases[i];
                int ri = peak;
                while (ri < rightBound && x[ri] > hEval) ri++;
                double rightIp;
                if (ri == peak && x[ri] <= hEval)
                {
                    rightIp = ri;
                }
                else
                {
                    int j1 = Math.Max(peak, ri - 1);
                    int j2 = Math.Min(ri, rightBound);
                    if (j2 == j1) rightIp = j2;
                    else
                    {
                        double y1 = x[j1], y2 = x[j2];
                        double denom = (y2 - y1);
                        double frac = denom == 0 ? 0.0 : (hEval - y1) / denom;
                        rightIp = j1 + frac;
                    }
                }

                leftIps[i] = leftIp;
                rightIps[i] = rightIp;
                widths[i] = rightIp - leftIp;
            }

            return (widths, widthHeights, leftIps, rightIps);
        }
    }
}
