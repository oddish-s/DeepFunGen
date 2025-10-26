import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';
import I18n from '../i18n.js';
import Native from '../native.js';

const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.mpg', '.mpeg', '.wmv']);
const PATH_SPLIT_REGEX = /[\/]/;

const staged = new Map();
let stagedList;
let modelSelect;
let addButton;
let browseFilesButton;
let dropZone;
let htmlFileInput;

const optionInputs = {};
const trackedInputs = new WeakSet();

const DEFAULT_OPTIONS = {
    smooth: 3,
    prominence: 0.1,
    minProminence: 0.0,
    maxSlope: 10.0,
    boostSlope: 7.0,
    minSlope: 2.0,
    merge: 120.0,
    centralDeviation: 0.03,
    fftDenoise: true,
    fftFrames: 10,
};

function normaliseKey(path) {
    return path.trim().replace(/\\/g, '/').toLowerCase();
}

function stageMetadata(path) {
    const normalised = path.replace(/\\/g, '/');
    const trimmed = normalised.replace(/\/+$/, '');
    const name = trimmed.split(PATH_SPLIT_REGEX).pop() || trimmed || path;
    return { key: normaliseKey(path), path, name };
}

function markDirty(input) {
    if (!input || trackedInputs.has(input)) return;
    trackedInputs.add(input);
}

function bindDirtyTracker(input) {
    if (!input || input.dataset?.dirtyTrackerBound) {
        return;
    }
    const handler = () => markDirty(input);
    const events = input.tagName === 'SELECT'
        ? ['change']
        : input.type === 'checkbox'
            ? ['change']
            : ['input', 'change'];
    events.forEach((event) => input.addEventListener(event, handler));
    input.dataset.dirtyTrackerBound = '1';
}

function registerOptionInput(key, elementId) {
    if (optionInputs[key]) {
        return optionInputs[key];
    }
    const element = document.getElementById(elementId);
    if (element) {
        optionInputs[key] = element;
        bindDirtyTracker(element);
    }
    return element;
}

function ensureElements() {
    if (!stagedList) stagedList = document.getElementById('staged-items');
    if (!modelSelect) modelSelect = document.getElementById('model-select');
    if (!addButton) addButton = document.getElementById('btn-add-to-queue');
    if (!browseFilesButton) browseFilesButton = document.getElementById('btn-browse-files');
    if (!dropZone) dropZone = document.getElementById('drop-zone');
    registerOptionInput('smooth', 'queue-opt-smooth');
    registerOptionInput('prominence', 'queue-opt-prominence');
    registerOptionInput('minProminence', 'queue-opt-min-prominence');
    registerOptionInput('maxSlope', 'queue-opt-max-slope');
    registerOptionInput('boostSlope', 'queue-opt-boost-slope');
    registerOptionInput('minSlope', 'queue-opt-min-slope');
    registerOptionInput('merge', 'queue-opt-merge');
    registerOptionInput('centralDeviation', 'queue-opt-central-dev');
    registerOptionInput('fftDenoise', 'queue-opt-fft-denoise');
    registerOptionInput('fftFrames', 'queue-opt-fft-frames');
    registerOptionInput('fftWindow', 'queue-opt-fft-window');
}

function toast(key, fallback, type = 'success', count) {
    let message = fallback;
    switch (key) {
        case 'select':
            message = I18n.t('queue.toast_select');
            break;
        case 'dialog':
            message = I18n.t('queue.toast_dialog');
            break;
        case 'added': {
            const template = I18n.t('queue.toast_added');
            const total = typeof count === 'number' ? count : staged.size || 0;
            message = template.replace('{count}', total);
            break;
        }
        case 'skipped':
            message = I18n.t('queue.toast_skipped');
            break;
        case 'unsupported':
            message = I18n.t('add.toast_unsupported');
            break;
        case 'drop_failed':
            message = I18n.t('add.toast_drop_failed');
            break;
        case 'drop_skipped':
            message = I18n.t('add.toast_drop_skipped');
            break;
        case 'browser_no_path':
            message = I18n.t('add.toast_browser_no_path');
            break;
        default:
            break;
    }
    showToast(message || fallback, type);
}

function renderStagedItems() {
    ensureElements();
    if (!stagedList) return;
    stagedList.innerHTML = '';
    if (addButton) {
        if (modelSelect?.disabled) {
            addButton.disabled = true;
        } else {
            addButton.disabled = staged.size === 0;
        }
    }
    if (staged.size === 0) {
        const empty = document.createElement('div');
        empty.className = 'staging__empty text-muted';
        empty.dataset.i18n = 'queue.no_staged';
        empty.textContent = 'No files staged.';
        stagedList.appendChild(empty);
        I18n.apply();
        return;
    }
    const fragment = document.createDocumentFragment();
    staged.forEach((item) => {
        const entry = document.createElement('div');
        entry.className = 'staging__item';

        const name = document.createElement('div');
        name.className = 'staging__name';
        name.textContent = item.name;
        entry.appendChild(name);

        const path = document.createElement('div');
        path.className = 'text-muted staging__path';
        path.textContent = item.path;
        entry.appendChild(path);

        const actions = document.createElement('div');
        actions.className = 'flex gap-sm';

        const removeButton = document.createElement('button');
        removeButton.className = 'button button--ghost';
        removeButton.dataset.i18n = 'queue.remove';
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', () => {
            staged.delete(item.key);
            renderStagedItems();
        });
        actions.appendChild(removeButton);

        entry.appendChild(actions);
        fragment.appendChild(entry);
    });
    stagedList.appendChild(fragment);
    I18n.apply();
}

function applyDefaultOptions() {
    ensureElements();
    const defaults = State.get('settings')?.default_postprocess || {};

    const setNumeric = (input, value, fallbackKey) => {
        if (!input || trackedInputs.has(input)) return;
        if (value === undefined || value === null || Number.isNaN(value)) {
            input.value = DEFAULT_OPTIONS[fallbackKey] ?? '';
        } else {
            input.value = value;
        }
    };

    const setOptional = (input, value) => {
        if (!input || trackedInputs.has(input)) return;
        if (value === undefined || value === null || value === '') {
            input.value = '';
        } else {
            input.value = value;
        }
    };

    const setBoolean = (input, value, fallbackKey) => {
        if (!input || trackedInputs.has(input)) return;
        if (value === undefined || value === null) {
            input.checked = Boolean(DEFAULT_OPTIONS[fallbackKey]);
        } else {
            input.checked = Boolean(value);
        }
    };

    setNumeric(optionInputs.smooth, defaults.smooth_window_frames, 'smooth');
    setNumeric(optionInputs.prominence, defaults.prominence_ratio, 'prominence');
    setNumeric(optionInputs.minProminence, defaults.min_prominence, 'minProminence');
    setNumeric(optionInputs.maxSlope, defaults.max_slope, 'maxSlope');
    setNumeric(optionInputs.boostSlope, defaults.boost_slope, 'boostSlope');
    setNumeric(optionInputs.minSlope, defaults.min_slope, 'minSlope');
    setNumeric(optionInputs.merge, defaults.merge_threshold_ms, 'merge');
    setNumeric(optionInputs.centralDeviation, defaults.central_deviation_threshold, 'centralDeviation');
    setBoolean(optionInputs.fftDenoise, defaults.fft_denoise, 'fftDenoise');
    setNumeric(optionInputs.fftFrames, defaults.fft_frames_per_component, 'fftFrames');
    setOptional(optionInputs.fftWindow, defaults.fft_window_frames);
}

function isSupportedVideo(path) {
    const lower = path.toLowerCase();
    const index = lower.lastIndexOf('.');
    if (index === -1) return false;
    return VIDEO_EXTENSIONS.has(lower.slice(index));
}

function considerPath(rawPath) {
    if (typeof rawPath !== 'string') return;
    const path = rawPath.trim();
    if (!path) return;
    const key = normaliseKey(path);
    if (staged.has(key)) {
        return;
    }
    if (!isSupportedVideo(path)) {
        toast('unsupported', 'Unsupported file type.', 'error');
        return;
    }
    staged.set(key, stageMetadata(path));
}

function stagePaths(paths) {
    if (!Array.isArray(paths) || paths.length === 0) {
        return 0;
    }
    let added = 0;
    paths.forEach((path) => {
        const before = staged.size;
        considerPath(path);
        if (staged.size > before) {
            added += 1;
        }
    });
    if (added > 0) {
        renderStagedItems();
    }
    return added;
}

function parseFileUri(entry) {
    if (!entry) return null;
    try {
        const url = new URL(entry.trim());
        if (url.protocol !== 'file:') {
            return null;
        }
        let pathname = decodeURIComponent(url.pathname || '');
        if (/^\/[A-Za-z]:/.test(pathname)) {
            pathname = pathname.slice(1);
        }
        return pathname.split("\\").join("/");
    } catch (error) {
        const trimmed = entry.trim();
        if (/^[A-Za-z]:\\/.test(trimmed)) {
            return trimmed;
        }
        return null;
    }
}

function pathsFromDrop(event) {
    const results = new Set();
    const dataTransfer = event?.dataTransfer || event?.originalEvent?.dataTransfer;
    if (!dataTransfer) {
        return [];
    }
    const extractPath = (file) => {
        if (!file) return null;
        if (typeof file.path === 'string' && file.path.length) return file.path;
        if (typeof file.pywebviewFullPath === 'string' && file.pywebviewFullPath.length) return file.pywebviewFullPath;
        if (typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.length) return file.webkitRelativePath;
        if (typeof file.name === 'string' && file.name.length && typeof file.fullPath === 'string') return file.fullPath;
        return null;
    };
    const files = Array.from(dataTransfer.files || []);
    files.forEach((file) => {
        const path = extractPath(file);
        if (path) {
            results.add(path);
        }
    });
    const appendFromString = (value) => {
        if (!value) return;
        value.split(/\r?\n/).forEach((line) => {
            const path = parseFileUri(line);
            if (path) {
                results.add(path);
            }
        });
    };
    appendFromString(dataTransfer.getData?.('text/uri-list'));
    appendFromString(dataTransfer.getData?.('text/plain'));
    return Array.from(results);
}
function getNumber(input, fallback) {
    if (!input) return fallback;
    const value = Number(input.value);
    return Number.isFinite(value) ? value : fallback;
}

function getOptionalInt(input) {
    if (!input) return undefined;
    const trimmed = input.value.trim();
    if (!trimmed) return undefined;
    const value = Number(trimmed);
    return Number.isFinite(value) ? Math.trunc(value) : undefined;
}

function getPostprocessOptions() {
    const options = {
        smooth_window_frames: Math.trunc(getNumber(optionInputs.smooth, 3)),
        prominence_ratio: getNumber(optionInputs.prominence, 0.1),
        min_prominence: getNumber(optionInputs.minProminence, 0.0),
        max_slope: getNumber(optionInputs.maxSlope, 10.0),
        boost_slope: getNumber(optionInputs.boostSlope, 7.0),
        min_slope: getNumber(optionInputs.minSlope, 2.0),
        merge_threshold_ms: getNumber(optionInputs.merge, 120.0),
        central_deviation_threshold: getNumber(optionInputs.centralDeviation, 0.03),
        fft_denoise: optionInputs.fftDenoise ? Boolean(optionInputs.fftDenoise.checked) : true,
        fft_frames_per_component: Math.trunc(getNumber(optionInputs.fftFrames, 10)),
    };
    const windowFrames = getOptionalInt(optionInputs.fftWindow);
    if (windowFrames !== undefined) {
        options.fft_window_frames = windowFrames;
    }
    return options;
}

function renderModels(models) {
    ensureElements();
    if (!modelSelect) return;
    modelSelect.innerHTML = '';
    if (!models || models.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No models found';
        modelSelect.appendChild(option);
        modelSelect.disabled = true;
        addButton?.setAttribute('disabled', 'disabled');
        return;
    }
    modelSelect.disabled = false;
    addButton?.removeAttribute('disabled');
    models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.display_name || model.name || model.path;
        modelSelect.appendChild(option);
    });
    const defaultModel = State.get('settings')?.default_model_path;
    if (defaultModel) {
        modelSelect.value = defaultModel;
    }
    renderStagedItems();
}

async function refreshQueue() {
    try {
        const data = await Api.get('/api/queue');
        State.setQueue(data || []);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function enqueueFiles() {
    ensureElements();
    if (staged.size === 0) {
        toast('select', 'Select at least one video.', 'error');
        return;
    }
    const payload = {
        video_paths: Array.from(staged.values()).map((item) => item.path),
        model_path: modelSelect?.value || undefined,
        postprocess_options: getPostprocessOptions(),
    };
    try {
        const result = await Api.post('/api/queue/add', payload);
        if (result?.added_count) {
            toast('added', 'Added to queue.', 'success', result.added_count);
            staged.clear();
            renderStagedItems();
            await refreshQueue();
        } else if (result?.skipped?.length) {
            toast('skipped', 'All files skipped.', 'error');
        }
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function attachFilePickers() {
    ensureElements();
    if (browseFilesButton) {
        browseFilesButton.addEventListener('click', async () => {
            const files = await Native.selectFiles();
            if (Array.isArray(files)) {
                stagePaths(files);
                return;
            }
            if (!Native.hasNativeDialogs()) {
                openHtmlFilePicker();
                return;
            }
            toast('dialog', 'File dialog not available.', 'error');
        });
    }
}

function attachDropZone() {
    ensureElements();
    if (!dropZone) return;
    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event?.dataTransfer) {
            event.dataTransfer.dropEffect = 'copy';
        }
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (event) => {
        dropZone.classList.remove('drag-over');
        const paths = pathsFromDrop(event);
        const added = stagePaths(paths);
        if (added > 0) {
            event.preventDefault();
            event.stopPropagation();
        }
    });
}

function subscribeState() {
    State.subscribe('models', renderModels);
    State.subscribe('settings', () => {
        renderModels(State.get('models') || []);
        applyDefaultOptions();
    });
}

export const AddView = {
    init() {
        ensureElements();
        attachFilePickers();
        attachDropZone();
        addButton?.addEventListener('click', enqueueFiles);
        subscribeState();
        renderModels(State.get('models') || []);
        applyDefaultOptions();
        renderStagedItems();
    },
    show() {
        // no-op
    },
};

export default AddView;

if (typeof window !== 'undefined') {
    window.addEventListener('native:files-dropped', (event) => {
        const detail = event?.detail;
        const paths = Array.isArray(detail) ? detail : [];
        const added = stagePaths(paths);
        if (paths.length && added === 0) {
            toast('drop_skipped', 'No new files were staged.', 'info');
        }
    });
}

function ensureHtmlPicker() {
    if (htmlFileInput) return htmlFileInput;
    htmlFileInput = document.createElement('input');
    htmlFileInput.type = 'file';
    htmlFileInput.multiple = true;
    htmlFileInput.accept = Array.from(VIDEO_EXTENSIONS).join(',');
    htmlFileInput.style.display = 'none';
    htmlFileInput.addEventListener('change', () => {
        const files = Array.from(htmlFileInput?.files || []);
        if (!files.length) return;
        const paths = [];
        const missingPaths = [];
        files.forEach((file) => {
            const path = typeof file?.path === 'string' && file.path ? file.path : null;
            if (path) {
                paths.push(path);
            } else {
                missingPaths.push(file.name);
            }
        });
        const added = stagePaths(paths);
        if (missingPaths.length) {
            toast('browser_no_path', 'Browser security blocks access to file paths. Please use the desktop app.', 'error');
        }
        if (added === 0 && !missingPaths.length) {
            toast('drop_skipped', 'No new files were staged.', 'info');
        }
        htmlFileInput.value = '';
    });
    document.body.appendChild(htmlFileInput);
    return htmlFileInput;
}

function openHtmlFilePicker() {
    const input = ensureHtmlPicker();
    if (!input) {
        toast('dialog', 'File dialog not available.', 'error');
        return;
    }
    input.value = '';
    input.click();
}

