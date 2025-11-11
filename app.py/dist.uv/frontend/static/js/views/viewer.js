import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';
import I18n from '../i18n.js';

const FRAME_WINDOW_FRAMES = 100;
const DEFAULT_FRAME_RATE = 30;
const PREVIEW_COOLDOWN_MS = 120;
const VALUE_MIN = 0;
const VALUE_MAX = 100;
const DEFAULT_POSTPROCESS = {
    smooth_window_frames: 3,
    prominence_ratio: 0.1,
    min_prominence: 0,
    max_slope: 10,
    boost_slope: 7,
    min_slope: 0,
    merge_threshold_ms: 120,
    fft_denoise: true,
    fft_frames_per_component: 10,
    fft_window_frames: null,
};

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
let optionsFormEl;
let optionsApplyButton;
let optionsStatusEl;
const optionInputs = {};
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
let isApplyingOptions = false;
let currentJobDetail = null;
let currentJobOptions = null;

function ensureElements() {
    videoNameEl = videoNameEl || document.getElementById('viewer-video-name');
    modelNameEl = modelNameEl || document.getElementById('viewer-model-name');
    cursorTimeEl = cursorTimeEl || document.getElementById('viewer-cursor-time');
    cursorValueEl = cursorValueEl || document.getElementById('viewer-cursor-value');
    windowLabelEl = windowLabelEl || document.getElementById('viewer-window-label');
    previewImageEl = previewImageEl || document.getElementById('viewer-preview');
    previewTimeEl = previewTimeEl || document.getElementById('viewer-preview-time');
    valueBarEl = valueBarEl || document.getElementById('viewer-value-bar');
    optionsFormEl = optionsFormEl || document.getElementById('viewer-options-form');
    optionsApplyButton = optionsApplyButton || document.getElementById('viewer-options-apply');
    optionsStatusEl = optionsStatusEl || document.getElementById('viewer-options-status');
    if (!optionInputs.smooth_window_frames) optionInputs.smooth_window_frames = document.getElementById('viewer-opt-smooth');
    if (!optionInputs.prominence_ratio) optionInputs.prominence_ratio = document.getElementById('viewer-opt-prominence');
    if (!optionInputs.min_prominence) optionInputs.min_prominence = document.getElementById('viewer-opt-min-prominence');
    if (!optionInputs.max_slope) optionInputs.max_slope = document.getElementById('viewer-opt-max-slope');
    if (!optionInputs.boost_slope) optionInputs.boost_slope = document.getElementById('viewer-opt-boost-slope');
    if (!optionInputs.min_slope) optionInputs.min_slope = document.getElementById('viewer-opt-min-slope');
    if (!optionInputs.merge_threshold_ms) optionInputs.merge_threshold_ms = document.getElementById('viewer-opt-merge');
    if (!optionInputs.fft_denoise) optionInputs.fft_denoise = document.getElementById('viewer-opt-fft-denoise');
    if (!optionInputs.fft_frames_per_component) optionInputs.fft_frames_per_component = document.getElementById('viewer-opt-fft-frames');
    if (!optionInputs.fft_window_frames) optionInputs.fft_window_frames = document.getElementById('viewer-opt-fft-window');
    backButtonEl = backButtonEl || document.getElementById('viewer-back-button');
    timelineEl = timelineEl || document.getElementById('viewer-chart-timeline');
    zoomPredEl = zoomPredEl || document.getElementById('viewer-chart-detail');
    zoomProcessedEl = zoomProcessedEl || document.getElementById('viewer-chart-processed');
}

function setOptionsStatusMessage(key, fallback = '') {
    ensureElements();
    if (!optionsStatusEl) return;
    if (!key) {
        optionsStatusEl.textContent = '';
        if (optionsStatusEl.dataset) {
            delete optionsStatusEl.dataset.i18n;
        }
        return;
    }
    if (optionsStatusEl.dataset) {
        optionsStatusEl.dataset.i18n = key;
    }
    const translated = I18n.t(key);
    optionsStatusEl.textContent = translated && translated !== key ? translated : fallback;
}

function setFormInputsDisabled(disabled) {
    Object.values(optionInputs).forEach((input) => {
        if (!input) return;
        input.disabled = Boolean(disabled);
    });
}

function populateOptions(options, available = false) {
    ensureElements();
    if (!available) {
        currentJobOptions = null;
        setFormInputsDisabled(true);
        Object.values(optionInputs).forEach((input) => {
            if (!input) return;
            if (input.type === 'checkbox') {
                input.checked = false;
            } else {
                input.value = '';
            }
        });
        setOptionsStatusMessage('viewer.options_status_select', '');
        if (optionsApplyButton) {
            optionsApplyButton.disabled = true;
        }
        return;
    }

    const source = {
        ...DEFAULT_POSTPROCESS,
        ...(options || {}),
    };

    setFormInputsDisabled(false);

    if (optionInputs.smooth_window_frames) optionInputs.smooth_window_frames.value = source.smooth_window_frames ?? DEFAULT_POSTPROCESS.smooth_window_frames;
    if (optionInputs.prominence_ratio) optionInputs.prominence_ratio.value = source.prominence_ratio ?? DEFAULT_POSTPROCESS.prominence_ratio;
    if (optionInputs.min_prominence) optionInputs.min_prominence.value = source.min_prominence ?? DEFAULT_POSTPROCESS.min_prominence;
    if (optionInputs.max_slope) optionInputs.max_slope.value = source.max_slope ?? DEFAULT_POSTPROCESS.max_slope;
    if (optionInputs.boost_slope) optionInputs.boost_slope.value = source.boost_slope ?? DEFAULT_POSTPROCESS.boost_slope;
    if (optionInputs.min_slope) optionInputs.min_slope.value = source.min_slope ?? DEFAULT_POSTPROCESS.min_slope;
    if (optionInputs.merge_threshold_ms) optionInputs.merge_threshold_ms.value = source.merge_threshold_ms ?? DEFAULT_POSTPROCESS.merge_threshold_ms;
    if (optionInputs.fft_denoise) optionInputs.fft_denoise.checked = Boolean(source.fft_denoise ?? DEFAULT_POSTPROCESS.fft_denoise);
    if (optionInputs.fft_frames_per_component) optionInputs.fft_frames_per_component.value = source.fft_frames_per_component ?? DEFAULT_POSTPROCESS.fft_frames_per_component;
    if (optionInputs.fft_window_frames) {
        if (source.fft_window_frames === null || source.fft_window_frames === undefined) {
            optionInputs.fft_window_frames.value = '';
        } else {
            optionInputs.fft_window_frames.value = source.fft_window_frames;
        }
    }

    currentJobOptions = {
        smooth_window_frames: Number(source.smooth_window_frames ?? DEFAULT_POSTPROCESS.smooth_window_frames),
        prominence_ratio: Number(source.prominence_ratio ?? DEFAULT_POSTPROCESS.prominence_ratio),
        min_prominence: Number(source.min_prominence ?? DEFAULT_POSTPROCESS.min_prominence),
        max_slope: Number(source.max_slope ?? DEFAULT_POSTPROCESS.max_slope),
        boost_slope: Number(source.boost_slope ?? DEFAULT_POSTPROCESS.boost_slope),
        min_slope: Number(source.min_slope ?? DEFAULT_POSTPROCESS.min_slope),
        merge_threshold_ms: Number(source.merge_threshold_ms ?? DEFAULT_POSTPROCESS.merge_threshold_ms),
        fft_denoise: Boolean(source.fft_denoise ?? DEFAULT_POSTPROCESS.fft_denoise),
        fft_frames_per_component: Number(source.fft_frames_per_component ?? DEFAULT_POSTPROCESS.fft_frames_per_component),
        fft_window_frames: source.fft_window_frames === null || source.fft_window_frames === undefined
            ? null
            : Number(source.fft_window_frames),
    };

    updateOptionsControls();
}

function readInt(input, fallback, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
    if (!input) return fallback;
    const value = Number(input.value);
    if (!Number.isFinite(value)) return fallback;
    const rounded = Math.round(value);
    return Math.min(Math.max(rounded, min), max);
}

function readFloat(input, fallback, min = -Infinity, max = Infinity) {
    if (!input) return fallback;
    const value = Number(input.value);
    if (!Number.isFinite(value)) return fallback;
    const clamped = Math.min(Math.max(value, min), max);
    return clamped;
}

function readOptionalInt(input, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
    if (!input) return null;
    const raw = input.value;
    if (raw === null || raw === undefined || raw === '') return null;
    const value = Number(raw);
    if (!Number.isFinite(value)) return null;
    const rounded = Math.round(value);
    return Math.min(Math.max(rounded, min), max);
}

function collectOptionValues() {
    if (!currentJobDetail) return null;
    return {
        smooth_window_frames: readInt(optionInputs.smooth_window_frames, DEFAULT_POSTPROCESS.smooth_window_frames, 1, 512),
        prominence_ratio: readFloat(optionInputs.prominence_ratio, DEFAULT_POSTPROCESS.prominence_ratio, 0, 1),
        min_prominence: readFloat(optionInputs.min_prominence, DEFAULT_POSTPROCESS.min_prominence, 0),
        max_slope: readFloat(optionInputs.max_slope, DEFAULT_POSTPROCESS.max_slope, 0),
        boost_slope: readFloat(optionInputs.boost_slope, DEFAULT_POSTPROCESS.boost_slope, 0),
        min_slope: readFloat(optionInputs.min_slope, DEFAULT_POSTPROCESS.min_slope, 0),
        merge_threshold_ms: readFloat(optionInputs.merge_threshold_ms, DEFAULT_POSTPROCESS.merge_threshold_ms, 0),
        fft_denoise: Boolean(optionInputs.fft_denoise?.checked ?? DEFAULT_POSTPROCESS.fft_denoise),
        fft_frames_per_component: readInt(optionInputs.fft_frames_per_component, DEFAULT_POSTPROCESS.fft_frames_per_component, 1, 10000),
        fft_window_frames: readOptionalInt(optionInputs.fft_window_frames, 1, 10000),
    };
}

function normaliseOptions(options) {
    if (!options) return null;
    return {
        smooth_window_frames: Number(options.smooth_window_frames),
        prominence_ratio: Number(options.prominence_ratio),
        min_prominence: Number(options.min_prominence),
        max_slope: Number(options.max_slope),
        boost_slope: Number(options.boost_slope),
        min_slope: Number(options.min_slope),
        merge_threshold_ms: Number(options.merge_threshold_ms),
        fft_denoise: Boolean(options.fft_denoise),
        fft_frames_per_component: Number(options.fft_frames_per_component),
        fft_window_frames: options.fft_window_frames === null || options.fft_window_frames === undefined
            ? null
            : Number(options.fft_window_frames),
    };
}

function optionsEqual(a, b) {
    const normalisedA = normaliseOptions(a);
    const normalisedB = normaliseOptions(b);
    if (!normalisedA || !normalisedB) return false;
    return JSON.stringify(normalisedA) === JSON.stringify(normalisedB);
}

function updateOptionsControls() {
    ensureElements();
    if (!optionsApplyButton) return;
    if (!currentJobDetail) {
        optionsApplyButton.disabled = true;
        setOptionsStatusMessage('viewer.options_status_select', '');
        return;
    }
    if (isApplyingOptions) {
        optionsApplyButton.disabled = true;
        setOptionsStatusMessage('viewer.options_status_busy', '');
        return;
    }
    const status = String(currentJobDetail.status || '').toLowerCase();
    if (status !== 'completed') {
        optionsApplyButton.disabled = true;
        setOptionsStatusMessage('viewer.options_status_unavailable', '');
        return;
    }
    optionsApplyButton.disabled = false;
    setOptionsStatusMessage('', '');
}

async function applyOptions(event) {
    if (event) {
        event.preventDefault();
    }
    if (!currentJobId || !currentJobDetail) {
        showToast(I18n.t('viewer.options_status_select') || 'Select a completed job first.', 'info');
        return;
    }
    const status = String(currentJobDetail.status || '').toLowerCase();
    if (status !== 'completed') {
        showToast(I18n.t('viewer.options_status_unavailable') || 'Wait for processing to finish.', 'info');
        return;
    }
    const nextOptions = collectOptionValues();
    if (!nextOptions) {
        showToast(I18n.t('viewer.options_toast_invalid') || 'Unable to gather options.', 'error');
        return;
    }
    if (currentJobOptions && optionsEqual(currentJobOptions, nextOptions)) {
        showToast(I18n.t('viewer.options_toast_nochange') || 'No option changes detected.', 'info');
        return;
    }
    isApplyingOptions = true;
    updateOptionsControls();
    setOptionsStatusMessage('viewer.options_status_busy', '');
    let succeeded = false;
    let failed = false;
    const previousWindowStart = windowStartMs;
    try {
        await Api.post(`/api/results/${currentJobId}/postprocess`, nextOptions);
        succeeded = true;
        await loadJob(currentJobId, { restoreWindowStart: previousWindowStart });
        try {
            const updatedQueue = await Api.get('/api/queue');
            if (Array.isArray(updatedQueue)) {
                State.setQueue(updatedQueue);
            }
        } catch (refreshError) {
            console.warn('Failed to refresh queue after reprocess', refreshError);
        }
        showToast(I18n.t('viewer.options_toast_applied') || 'Options updated.', 'success');
    } catch (error) {
        console.error('apply options failed', error);
        failed = true;
        showToast(error.message, 'error');
    } finally {
        isApplyingOptions = false;
        updateOptionsControls();
        if (succeeded) {
            setOptionsStatusMessage('viewer.options_status_applied', '');
        } else if (failed) {
            setOptionsStatusMessage('viewer.options_status_error', '');
        }
    }
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
    let hasFinite = false;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let idx = 0; idx < values.length; idx += 1) {
        const value = values[idx];
        if (typeof value !== 'number' || !Number.isFinite(value)) {
            continue;
        }
        hasFinite = true;
        if (value < min) min = value;
        if (value > max) max = value;
    }
    if (!hasFinite) {
        return [fallbackMin, fallbackMax];
    }
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
    if (!timelineEl) return Promise.resolve();
    const plotly = window.Plotly;
    if (!plotly || typeof plotly.react !== 'function') {
        showChartPlaceholder(timelineEl);
        return Promise.resolve();
    }
    if (!timestamps.length || !predictedChanges.length) {
        showChartPlaceholder(timelineEl);
        return Promise.resolve();
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
    return plotly.react(timelineEl, data, layout, config)
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

function setWindowStart(start, { forceRefresh = false } = {}) {
    if (windowDurationMs <= 0) return;
    const clampedStart = clampWindowStart(start);
    if (!forceRefresh && Math.abs(clampedStart - windowStartMs) < 0.5) {
        windowStartMs = clampedStart;
        const [rangeStart, rangeEnd] = getWindowRange();
        updateWindowLabel(rangeStart, rangeEnd);
        updateDetailCharts();
        updateTimelineSelection();
        return;
    }
    windowStartMs = clampedStart;
    const [rangeStart, rangeEnd] = getWindowRange();
    updateWindowLabel(rangeStart, rangeEnd);
    updateDetailCharts();
    updateTimelineSelection();
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
    currentJobDetail = null;
    currentJobOptions = null;
    isApplyingOptions = false;
    populateOptions(null, false);
    updateOptionsControls();
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

async function loadJob(jobId, options = {}) {
    ensureElements();
    currentJobId = jobId;
    const restoreWindowStart =
        options && Number.isFinite(options.restoreWindowStart)
            ? Number(options.restoreWindowStart)
            : null;
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
        currentJobDetail = job;
        populateOptions(job.postprocess_options, true);
        updateOptionsControls();

        loadPredictions(data.predictions || {});
        currentFrameRate = Number(job.frame_rate) > 0 ? Number(job.frame_rate) : DEFAULT_FRAME_RATE;
        windowDurationMs = computeWindowDuration(currentFrameRate, maxTimelineMs);

        if (!timestamps.length) {
            windowStartMs = 0;
            updateWindowLabel(NaN, NaN);
            resetCharts();
            return;
        }

        const initialWindowStart = restoreWindowStart !== null ? restoreWindowStart : 0;
        windowStartMs = clampWindowStart(initialWindowStart);

        await drawTimelineChart();
        setWindowStart(initialWindowStart, { forceRefresh: true });
    } catch (error) {
        showToast(error.message, 'error');
        resetViewer();
    }
}

export const ViewerView = {
    init() {
        ensureElements();
        resetViewer();
        if (optionsFormEl && !optionsFormEl.dataset.submitBound) {
            optionsFormEl.addEventListener('submit', applyOptions);
            optionsFormEl.dataset.submitBound = '1';
        }
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
