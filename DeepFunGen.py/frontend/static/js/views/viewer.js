import Api from '../api.js';
import { showToast } from '../app.js';
import I18n from '../i18n.js';

const FRAME_WINDOW_FRAMES = 100;
const DEFAULT_FRAME_RATE = 30;
const PREVIEW_COOLDOWN_MS = 120;
const VALUE_MIN = 0;
const VALUE_MAX = 100;

let currentJobId = null;
let timestamps = [];
let predictedChanges = [];
let processedValues = [];
let fftValues = [];
let phaseMarker = [];
let phaseSource = [];
let windowStartMs = 0;
let windowDurationMs = 0;
let maxTimelineMs = 0;
let currentFrameRate = DEFAULT_FRAME_RATE;

let videoNameEl;
let modelNameEl;
let cursorTimeEl;
let cursorValueEl;
let windowLabelEl;
let previewImageEl;
let previewTimeEl;
let valueBarEl;
let optionsListEl;
let backButtonEl;
let timelineEl;
let zoomPredEl;
let zoomProcessedEl;

let lastPreviewTimestamp = -Infinity;
let currentPreviewUrl = null;
let previewController = null;
let isTimelineUpdating = false;

let changeGlobalRange = [-10, 10];
let processedGlobalRange = [VALUE_MIN, VALUE_MAX];

function ensureElements() {
    videoNameEl = videoNameEl || document.getElementById('viewer-video-name');
    modelNameEl = modelNameEl || document.getElementById('viewer-model-name');
    cursorTimeEl = cursorTimeEl || document.getElementById('viewer-cursor-time');
    cursorValueEl = cursorValueEl || document.getElementById('viewer-cursor-value');
    windowLabelEl = windowLabelEl || document.getElementById('viewer-window-label');
    previewImageEl = previewImageEl || document.getElementById('viewer-preview');
    previewTimeEl = previewTimeEl || document.getElementById('viewer-preview-time');
    valueBarEl = valueBarEl || document.getElementById('viewer-value-bar');
    optionsListEl = optionsListEl || document.getElementById('viewer-options-list');
    backButtonEl = backButtonEl || document.getElementById('viewer-back-button');
    timelineEl = timelineEl || document.getElementById('viewer-chart-timeline');
    zoomPredEl = zoomPredEl || document.getElementById('viewer-chart-detail');
    zoomProcessedEl = zoomProcessedEl || document.getElementById('viewer-chart-processed');
}

function purgeChart(element) {
    if (!element || !window.Plotly || typeof window.Plotly.purge !== 'function') return;
    try {
        window.Plotly.purge(element);
    } catch (error) {
        console.warn('plotly purge failed', error);
    }
}

function clearChartElement(element) {
    if (!element) return;
    purgeChart(element);
    element.innerHTML = '';
}

function showChartPlaceholder(element) {
    if (!element) return;
    purgeChart(element);
    element.innerHTML = '';
    const placeholder = document.createElement('div');
    placeholder.className = 'viewer-chart__placeholder';
    placeholder.dataset.i18n = 'viewer.no_data';
    placeholder.textContent = 'Select a completed job to visualise predictions.';
    element.appendChild(placeholder);
    I18n.apply();
}

function revokePreviewUrl() {
    if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
        currentPreviewUrl = null;
    }
}

function resetCharts() {
    ensureElements();
    showChartPlaceholder(timelineEl);
    showChartPlaceholder(zoomPredEl);
    showChartPlaceholder(zoomProcessedEl);
}

function updateValueBar(value) {
    ensureElements();
    if (!valueBarEl) return;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        valueBarEl.style.width = '0%';
        valueBarEl.dataset.value = '0';
        return;
    }
    const clamped = Math.max(VALUE_MIN, Math.min(VALUE_MAX, numeric));
    valueBarEl.style.width = `${clamped}%`;
    valueBarEl.dataset.value = String(Math.round(clamped));
}

function updateWindowLabel(start, end) {
    ensureElements();
    if (!windowLabelEl) return;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        windowLabelEl.textContent = '-';
        return;
    }
    const fmt = (ms) => (ms / 1000).toFixed(2).replace(/\.00$/, '');
    windowLabelEl.textContent = `${fmt(start)}s – ${fmt(end)}s`;
}

function formatTimestamp(ms) {
    if (!Number.isFinite(ms)) return '-';
    const totalSeconds = Math.max(0, ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds - minutes * 60;
    return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(2).padStart(5, '0')}`;
}

async function fetchPreviewFrame(jobId, timestamp) {
    ensureElements();
    if (!previewImageEl || !jobId) return;
    if (Math.abs(timestamp - lastPreviewTimestamp) < PREVIEW_COOLDOWN_MS) return;
    lastPreviewTimestamp = timestamp;

    if (previewController) {
        previewController.abort();
    }
    previewController = new AbortController();

    try {
        const safeTimestamp = Math.max(0, Math.round(timestamp));
        const response = await fetch(
            `/api/results/${jobId}/frame?timestamp_ms=${safeTimestamp}`,
            { signal: previewController.signal },
        );
        if (!response.ok) {
            throw new Error(`Failed to fetch frame (${response.status})`);
        }
        const blob = await response.blob();
        revokePreviewUrl();
        currentPreviewUrl = URL.createObjectURL(blob);
        previewImageEl.src = currentPreviewUrl;
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.warn('preview fetch failed', error);
        }
    }
}

function handleHover(event) {
    if (!event?.points?.length) return;
    const point = event.points[0];
    const timestamp = Number(point.x) || 0;
    if (cursorTimeEl) cursorTimeEl.textContent = `${Math.round(timestamp)} ms`;
    if (previewTimeEl) previewTimeEl.textContent = formatTimestamp(timestamp);

    const valuePoint =
        event.points.find((pt) => pt.data?.meta === 'processed') ||
        event.points.find((pt) => pt.data?.meta === 'predicted_change');
    const numericValue = valuePoint ? Number(valuePoint.y) : Number(point.y);
    if (cursorValueEl) cursorValueEl.textContent = Number.isFinite(numericValue) ? `${Math.round(numericValue)}` : '-';
    updateValueBar(numericValue);

    if (currentJobId) {
        fetchPreviewFrame(currentJobId, timestamp);
    }
}

function bindHoverHandler(element) {
    element?.removeAllListeners?.('plotly_hover');
    element?.on?.('plotly_hover', handleHover);
}

function computeWindowDuration(frameRate, totalDuration) {
    const rate = Number(frameRate) > 0 ? Number(frameRate) : DEFAULT_FRAME_RATE;
    const ideal = (FRAME_WINDOW_FRAMES / rate) * 1000;
    if (Number(totalDuration) > 0) {
        return Math.max(1, Math.min(ideal, Number(totalDuration)));
    }
    return Math.max(1, ideal);
}

function computeAxisRange(values, fallbackMin = VALUE_MIN, fallbackMax = VALUE_MAX, paddingRatio = 0.05) {
    if (!Array.isArray(values) || !values.length) {
        return [fallbackMin, fallbackMax];
    }
    const finite = values.filter((value) => typeof value === 'number' && Number.isFinite(value));
    if (!finite.length) {
        return [fallbackMin, fallbackMax];
    }
    let min = Math.min(...finite);
    let max = Math.max(...finite);
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return [fallbackMin, fallbackMax];
    }
    if (min === max) {
        const offset = Math.max(Math.abs(min) * paddingRatio, 0.5);
        return [min - offset, max + offset];
    }
    const span = max - min;
    const padding = Math.max(span * paddingRatio, 0.5);
    return [min - padding, max + padding];
}

function clampWindowStart(start) {
    const duration = Math.max(1, windowDurationMs);
    const maxStart = Math.max(0, maxTimelineMs - duration);
    if (!Number.isFinite(start)) {
        return Math.min(Math.max(0, windowStartMs), maxStart);
    }
    return Math.min(Math.max(0, start), maxStart);
}

function getWindowRange() {
    const start = Math.max(0, windowStartMs);
    const end = start + Math.max(1, windowDurationMs);
    return [start, end];
}

function getWindowIndices(start, end) {
    if (!timestamps.length) return [];
    const indices = timestamps.reduce((acc, ts, idx) => {
        if (ts >= start && ts <= end) {
            acc.push(idx);
        }
        return acc;
    }, []);

    if (!indices.length) {
        const nextIndex = timestamps.findIndex((ts) => ts > start);
        if (nextIndex === -1) {
            indices.push(timestamps.length - 1);
        } else if (nextIndex === 0) {
            indices.push(0);
        } else {
            indices.push(nextIndex - 1, nextIndex);
        }
    } else {
        const first = indices[0];
        if (first > 0) {
            indices.unshift(first - 1);
        }
        const last = indices[indices.length - 1];
        if (last < timestamps.length - 1) {
            indices.push(last + 1);
        }
    }

    return Array.from(new Set(indices)).sort((a, b) => a - b);
}

function sliceSeries(series, indices) {
    if (!Array.isArray(series)) return [];
    return indices.map((index) => series[index]);
}

function buildTimelineShape() {
    if (!timestamps.length || windowDurationMs <= 0) return [];
    const [start, end] = getWindowRange();
    return [
        {
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: start,
            x1: end,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(91, 140, 255, 0.15)',
            line: { color: 'rgba(91, 140, 255, 0.85)', width: 2 },
            layer: 'below',
        },
    ];
}

function drawTimelineChart() {
    ensureElements();
    if (!timelineEl) return;
    const plotly = window.Plotly;
    if (!plotly || typeof plotly.react !== 'function') {
        showChartPlaceholder(timelineEl);
        return;
    }
    if (!timestamps.length || !predictedChanges.length) {
        showChartPlaceholder(timelineEl);
        return;
    }

    clearChartElement(timelineEl);

    const timelineExtent = Math.max(maxTimelineMs, windowDurationMs || 1);
    const yRange = computeAxisRange(predictedChanges, changeGlobalRange[0], changeGlobalRange[1], 0.08);
    changeGlobalRange = yRange.slice();
    const data = [
        {
            x: timestamps,
            y: predictedChanges,
            name: 'Predicted Δ',
            meta: 'predicted_change',
            mode: 'lines',
            line: { color: '#5b8cff', width: 1.8 },
            hovertemplate: 'Change %{y:.2f}<extra></extra>',
        },
    ];

    const layout = {
        margin: { t: 20, r: 16, b: 36, l: 50 },
        height: 200,
        hovermode: 'x',
        dragmode: 'pan',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Timeline (ms)',
            range: [0, timelineExtent],
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
        yaxis: {
            title: 'Change',
            range: yRange,
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
        shapes: buildTimelineShape(),
    };

    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
        modeBarButtonsToRemove: [
            'zoom2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d',
            'lasso2d',
            'select2d',
            'pan2d',
        ],
        edits: { shapePosition: true, shapeSize: false },
    };

    isTimelineUpdating = true;
    plotly.react(timelineEl, data, layout, config)
        .then(() => {
            isTimelineUpdating = false;
            attachTimelineHandlers();
        })
        .catch((error) => {
            isTimelineUpdating = false;
            console.error('timeline render failed', error);
        });
}

function updateTimelineSelection() {
    if (!timelineEl || !window.Plotly || typeof window.Plotly.relayout !== 'function') return;
    if (!timestamps.length || windowDurationMs <= 0) return;
    const [start, end] = getWindowRange();
    isTimelineUpdating = true;
    window.Plotly.relayout(timelineEl, {
        'shapes[0].x0': start,
        'shapes[0].x1': end,
        'shapes[0].y0': 0,
        'shapes[0].y1': 1,
    })
        .catch((error) => {
            console.warn('timeline relayout failed', error);
        })
        .finally(() => {
            isTimelineUpdating = false;
        });
}

function handleTimelineRelayout(eventData) {
    if (isTimelineUpdating) return;
    if (!eventData) return;
    const yMoved =
        Object.prototype.hasOwnProperty.call(eventData, 'shapes[0].y0') ||
        Object.prototype.hasOwnProperty.call(eventData, 'shapes[0].y1');
    let nextStart = null;
    if (Object.prototype.hasOwnProperty.call(eventData, 'shapes[0].x0')) {
        const candidate = Number(eventData['shapes[0].x0']);
        if (Number.isFinite(candidate)) {
            nextStart = candidate;
        }
    } else if (Object.prototype.hasOwnProperty.call(eventData, 'shapes[0].x1')) {
        const candidate = Number(eventData['shapes[0].x1']);
        if (Number.isFinite(candidate)) {
            nextStart = candidate - windowDurationMs;
        }
    }
    if (yMoved) {
        updateTimelineSelection();
    }
    if (nextStart === null || !Number.isFinite(nextStart)) return;
    setWindowStart(nextStart);
}

function handleTimelineClick(eventData) {
    if (!eventData?.points?.length) return;
    const point = eventData.points[0];
    const target = Number(point.x) - windowDurationMs / 2;
    if (!Number.isFinite(target)) return;
    setWindowStart(target);
}

function attachTimelineHandlers() {
    if (!timelineEl?.on) return;
    timelineEl.removeAllListeners?.('plotly_relayout');
    timelineEl.removeAllListeners?.('plotly_click');
    timelineEl.on('plotly_relayout', handleTimelineRelayout);
    timelineEl.on('plotly_click', handleTimelineClick);
    bindHoverHandler(timelineEl);
}

function drawZoomPredictedChart() {
    ensureElements();
    if (!zoomPredEl) return;
    const plotly = window.Plotly;
    if (!plotly || typeof plotly.react !== 'function') {
        showChartPlaceholder(zoomPredEl);
        return;
    }
    if (!timestamps.length || !predictedChanges.length) {
        showChartPlaceholder(zoomPredEl);
        return;
    }
    clearChartElement(zoomPredEl);
    const [start, end] = getWindowRange();
    const indices = getWindowIndices(start, end);
    const windowTimestamps = sliceSeries(timestamps, indices);
    const windowPredicted = sliceSeries(predictedChanges, indices).map((value) =>
        typeof value === 'number' && Number.isFinite(value) ? value : null,
    );
    const windowPhase = sliceSeries(phaseMarker, indices);
    const windowPhaseSource = sliceSeries(phaseSource, indices);
    const windowFft = sliceSeries(fftValues, indices).map((value) =>
        typeof value === 'number' && Number.isFinite(value) ? value : null,
    );

    const finitePredicted = windowPredicted.filter((value) => typeof value === 'number' && Number.isFinite(value));
    const finiteFft = windowFft.filter((value) => typeof value === 'number' && Number.isFinite(value));
    const rangeInputs = finitePredicted.length || finiteFft.length ? [...finitePredicted, ...finiteFft] : finitePredicted;
    const detailRange = computeAxisRange(rangeInputs, changeGlobalRange[0], changeGlobalRange[1], 0.05);

    const data = [
        {
            x: windowTimestamps,
            y: windowPredicted,
            name: 'Change',
            meta: 'predicted_change',
            mode: 'lines',
            line: { color: '#5b8cff', width: 2 },
            hovertemplate: 'Change %{y:.2f}<extra></extra>',
        },
    ];

    if (finiteFft.length) {
        data.push({
            x: windowTimestamps,
            y: windowFft,
            name: 'FFT Denoised',
            meta: 'fft',
            mode: 'lines',
            line: { color: '#9aa4c2', width: 1.6, dash: 'dot' },
            hovertemplate: 'FFT %{y:.3f}<extra></extra>',
        });
    }

    const peakX = [];
    const peakY = [];
    const peakText = [];
    const troughX = [];
    const troughY = [];
    const troughText = [];

    windowPhase.forEach((marker, idx) => {
        const markerValue = windowPhase[idx];
        if (typeof markerValue !== 'number' || !Number.isFinite(markerValue)) return;
        const ts = windowTimestamps[idx];
        const changeValue = windowPredicted[idx];
        if (typeof changeValue !== 'number' || !Number.isFinite(changeValue)) return;
        const source = windowPhaseSource[idx] || '';
        if (markerValue >= 99) {
            peakX.push(ts);
            peakY.push(changeValue);
            peakText.push(source || 'Peak');
        } else if (markerValue <= 1) {
            troughX.push(ts);
            troughY.push(changeValue);
            troughText.push(source || 'Trough');
        }
    });

    if (peakX.length) {
        data.push({
            x: peakX,
            y: peakY,
            text: peakText,
            name: 'Peaks',
            meta: 'peak',
            mode: 'markers',
            marker: { color: '#ff6b9a', size: 9, symbol: 'star' },
            hovertemplate: '%{text}<extra>Peak</extra>',
        });
    }
    if (troughX.length) {
        data.push({
            x: troughX,
            y: troughY,
            text: troughText,
            name: 'Troughs',
            meta: 'peak',
            mode: 'markers',
            marker: { color: '#67d5b5', size: 9, symbol: 'triangle-down' },
            hovertemplate: '%{text}<extra>Trough</extra>',
        });
    }

    const layout = {
        margin: { t: 26, r: 16, b: 36, l: 50 },
        height: 260,
        hovermode: 'x unified',
        legend: { orientation: 'h', x: 0, y: 1.1 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Time (ms)',
            range: [start, end],
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
        yaxis: {
            title: 'Change / FFT',
            range: detailRange,
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
    };

    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
        modeBarButtonsToRemove: [
            'zoom2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d',
            'pan2d',
            'lasso2d',
            'select2d',
        ],
    };

    plotly.react(zoomPredEl, data, layout, config)
        .then(() => {
            bindHoverHandler(zoomPredEl);
        })
        .catch((error) => {
            console.error('detail render failed', error);
        });
}

function drawProcessedChart() {
    ensureElements();
    if (!zoomProcessedEl) return;
    const plotly = window.Plotly;
    if (!plotly || typeof plotly.react !== 'function') {
        showChartPlaceholder(zoomProcessedEl);
        return;
    }
    if (!timestamps.length || !processedValues.length) {
        showChartPlaceholder(zoomProcessedEl);
        return;
    }
    clearChartElement(zoomProcessedEl);
    const [start, end] = getWindowRange();
    const indices = getWindowIndices(start, end);
    const windowTimestamps = sliceSeries(timestamps, indices);
    const windowProcessed = sliceSeries(processedValues, indices).map((value) =>
        typeof value === 'number' && Number.isFinite(value) ? value : null,
    );

    const rangeInputs = windowProcessed.filter((value) => typeof value === 'number' && Number.isFinite(value));
    const processedRange = computeAxisRange(rangeInputs, processedGlobalRange[0], processedGlobalRange[1], 0.05);

    const data = [
        {
            x: windowTimestamps,
            y: windowProcessed,
            name: 'Processed',
            meta: 'processed',
            mode: 'lines',
            line: { color: '#ffa600', width: 2.2 },
            hovertemplate: 'Processed %{y:.0f}<extra></extra>',
        },
    ];

    const layout = {
        margin: { t: 24, r: 16, b: 32, l: 50 },
        height: 220,
        hovermode: 'x',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Time (ms)',
            range: [start, end],
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
        yaxis: {
            title: 'Processed Value',
            range: processedRange,
            fixedrange: true,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.08)',
        },
    };

    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
        modeBarButtonsToRemove: [
            'zoom2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d',
            'pan2d',
            'lasso2d',
            'select2d',
        ],
    };

    plotly.react(zoomProcessedEl, data, layout, config)
        .then(() => {
            bindHoverHandler(zoomProcessedEl);
        })
        .catch((error) => {
            console.error('processed render failed', error);
        });
}

function updateDetailCharts() {
    if (!timestamps.length) {
        showChartPlaceholder(zoomPredEl);
        showChartPlaceholder(zoomProcessedEl);
        return;
    }
    drawZoomPredictedChart();
    drawProcessedChart();
}

function setWindowStart(start) {
    if (windowDurationMs <= 0) return;
    const clampedStart = clampWindowStart(start);
    if (Math.abs(clampedStart - windowStartMs) < 0.5) {
        updateTimelineSelection();
        return;
    }
    windowStartMs = clampedStart;
    const [rangeStart, rangeEnd] = getWindowRange();
    updateWindowLabel(rangeStart, rangeEnd);
    updateDetailCharts();
    updateTimelineSelection();
}

function renderOptions(options) {
    ensureElements();
    if (!optionsListEl) return;
    optionsListEl.innerHTML = '';
    if (!options || typeof options !== 'object') {
        const empty = document.createElement('div');
        empty.className = 'text-muted viewer-options__empty';
        empty.dataset.i18n = 'viewer.no_options';
        empty.textContent = 'No post-processing options.';
        optionsListEl.appendChild(empty);
        I18n.apply();
        return;
    }

    const entries = [
        ['smooth_window_frames', 'queue.opt_smooth'],
        ['prominence_ratio', 'queue.opt_prominence'],
        ['min_prominence', 'add.opt_min_prominence'],
        ['max_slope', 'add.opt_max_slope'],
        ['boost_slope', 'add.opt_boost_slope'],
        ['min_slope', 'add.opt_min_slope'],
        ['merge_threshold_ms', 'queue.opt_merge'],
        ['central_deviation_threshold', 'add.opt_central_dev'],
        ['fft_denoise', 'add.opt_fft_denoise'],
        ['fft_frames_per_component', 'add.opt_fft_frames'],
        ['fft_window_frames', 'add.opt_fft_window'],
    ];

    entries.forEach(([key, labelKey]) => {
        const value = options[key];
        if (value === undefined) return;
        const dt = document.createElement('dt');
        dt.textContent = I18n.t(labelKey);
        const dd = document.createElement('dd');
        if (value === null) {
            dd.textContent = I18n.t('viewer.auto');
        } else if (typeof value === 'boolean') {
            dd.textContent = I18n.t(value ? 'viewer.boolean_on' : 'viewer.boolean_off');
        } else if (typeof value === 'number') {
            const digits = Number.isInteger(value) ? 0 : 3;
            dd.textContent = Number(value)
                .toFixed(digits)
                .replace(/\.0+$/, '')
                .replace(/(\.\d+?)0+$/, '$1');
        } else {
            dd.textContent = String(value);
        }
        optionsListEl.appendChild(dt);
        optionsListEl.appendChild(dd);
    });
}

function resetViewer() {
    ensureElements();
    currentJobId = null;
    timestamps = [];
    predictedChanges = [];
    processedValues = [];
    fftValues = [];
    phaseMarker = [];
    phaseSource = [];
    windowStartMs = 0;
    windowDurationMs = 0;
    maxTimelineMs = 0;
    currentFrameRate = DEFAULT_FRAME_RATE;
    lastPreviewTimestamp = -Infinity;
    changeGlobalRange = [-10, 10];
    processedGlobalRange = [VALUE_MIN, VALUE_MAX];

    if (previewController) {
        previewController.abort();
        previewController = null;
    }
    revokePreviewUrl();
    resetCharts();
    if (videoNameEl) videoNameEl.textContent = '-';
    if (modelNameEl) modelNameEl.textContent = '-';
    if (cursorTimeEl) cursorTimeEl.textContent = '-';
    if (cursorValueEl) cursorValueEl.textContent = '-';
    if (previewTimeEl) previewTimeEl.textContent = '-';
    if (previewImageEl) previewImageEl.removeAttribute('src');
    updateValueBar(0);
    updateWindowLabel(NaN, NaN);
    renderOptions(null);
}

function normaliseNumericList(values) {
    if (!Array.isArray(values)) return [];
    return values.map((value) => {
        if (value === null || value === undefined) return null;
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : null;
    });
}

function loadPredictions(predictions) {
    timestamps = normaliseNumericList(predictions.timestamps || []).map((value) => value ?? 0);
    predictedChanges = normaliseNumericList(predictions.predicted_change || []);
    processedValues = normaliseNumericList(predictions.processed_value || []);
    fftValues = normaliseNumericList(predictions.fft_denoised || []);
    phaseMarker = normaliseNumericList(predictions.phase_marker || []);
    phaseSource = Array.isArray(predictions.phase_source) ? predictions.phase_source.slice() : [];

    changeGlobalRange = computeAxisRange(predictedChanges, -10, 10, 0.08);
    processedGlobalRange = computeAxisRange(processedValues, VALUE_MIN, VALUE_MAX, 0.05);

    if (timestamps.length) {
        maxTimelineMs = Number(timestamps[timestamps.length - 1]) || 0;
    } else {
        maxTimelineMs = 0;
    }
}

async function loadJob(jobId) {
    ensureElements();
    currentJobId = jobId;
    if (!jobId) {
        resetViewer();
        return;
    }
    try {
        const data = await Api.get(`/api/results/${jobId}`);
        const job = data.job || {};
        if (videoNameEl) videoNameEl.textContent = job.video_name || job.video_path || '-';
        if (modelNameEl) modelNameEl.textContent = job.model_path || '-';
        if (cursorTimeEl) cursorTimeEl.textContent = '-';
        if (cursorValueEl) cursorValueEl.textContent = '-';
        if (previewTimeEl) previewTimeEl.textContent = '-';
        updateValueBar(0);
        renderOptions(job.postprocess_options);

        loadPredictions(data.predictions || {});
        currentFrameRate = Number(job.frame_rate) > 0 ? Number(job.frame_rate) : DEFAULT_FRAME_RATE;
        windowDurationMs = computeWindowDuration(currentFrameRate, maxTimelineMs);
        windowStartMs = 0;
        const [rangeStart, rangeEnd] = getWindowRange();
        updateWindowLabel(rangeStart, rangeEnd);

        if (!timestamps.length) {
            resetCharts();
            return;
        }

        drawTimelineChart();
        updateDetailCharts();
    } catch (error) {
        showToast(error.message, 'error');
        resetViewer();
    }
}

export const ViewerView = {
    init() {
        ensureElements();
        resetViewer();
        backButtonEl?.addEventListener('click', () => {
            document.dispatchEvent(
                new CustomEvent('app:navigate', {
                    detail: { view: 'queue' },
                }),
            );
        });
    },
    show(payload = {}) {
        if (payload.jobId) {
            loadJob(payload.jobId);
        } else if (currentJobId) {
            loadJob(currentJobId);
        }
    },
};

export default ViewerView;
