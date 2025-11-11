import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';
import I18n from '../i18n.js';
import Native from '../native.js';

const PATH_SPLIT_REGEX = /[\\/]/;
const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.mpg', '.mpeg', '.wmv']);
const CANCELLABLE_STATUSES = new Set(['pending', 'processing']);
const AUTO_REFRESH_MS = 10000;
const STATUS_LABELS = {
    pending: 'queue.status.pending',
    processing: 'queue.status.processing',
    completed: 'queue.status.completed',
    failed: 'queue.status.failed',
    cancelled: 'queue.status.cancelled',
};

let queueTableBody;
let refreshButton;
let clearButton;
let dropZone;
let browseFilesButton;
let manualPathInput;
let manualPathButton;
let htmlFileInput;
let modelSelect;
let modelFilterToggle;
let modelFilterVrOnly = false;
let enqueueBusy = false;
let refreshTimer = null;

function ensureElements() {
    if (!queueTableBody) queueTableBody = document.querySelector('#queue-table tbody');
    if (!refreshButton) refreshButton = document.getElementById('btn-refresh-queue');
    if (!clearButton) clearButton = document.getElementById('btn-clear-finished');
    if (!dropZone) dropZone = document.getElementById('drop-zone');
    if (!browseFilesButton) browseFilesButton = document.getElementById('btn-browse-files');
    if (!manualPathInput) manualPathInput = document.getElementById('manual-path-input');
    if (!manualPathButton) manualPathButton = document.getElementById('btn-add-manual-path');
    if (!modelSelect) modelSelect = document.getElementById('model-select');
    if (!modelFilterToggle) modelFilterToggle = document.getElementById('model-filter-vr');
}

function toast(key, fallback, type = 'success', count) {
    let message = fallback;
    switch (key) {
        case 'cancelled':
            message = I18n.t('queue.toast_cancelled');
            break;
        case 'cleared':
            message = I18n.t('queue.toast_cleared');
            break;
        case 'added': {
            const template = I18n.t('queue.toast_added');
            const total = typeof count === 'number' ? count : 0;
            message = (typeof fallback === 'string' && fallback ? fallback : template).replace('{count}', total);
            break;
        }
        case 'skipped':
            message = I18n.t('queue.toast_skipped');
            type = 'info';
            break;
        case 'unsupported':
            message = I18n.t('add.toast_unsupported');
            type = 'error';
            break;
        case 'dialog':
            message = I18n.t('queue.toast_dialog');
            type = 'error';
            break;
        case 'drop_failed':
            message = I18n.t('add.toast_drop_failed');
            type = 'error';
            break;
        case 'drop_skipped':
            message = I18n.t('add.toast_drop_skipped');
            type = 'info';
            break;
        case 'browser_no_path':
            message = I18n.t('add.toast_browser_no_path');
            type = 'error';
            break;
        default:
            message = fallback;
    }
    showToast(message || fallback, type);
}

function formatProgress(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) return '0%';
    return `${Math.max(0, Math.min(100, Math.round(value * 100)))}%`;
}

function formatDecimal(value, digits = 2) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '0';
    const fixed = number.toFixed(digits);
    return fixed.replace(/\.0+$/, '').replace(/(\.\d+?)0+$/, '$1');
}

function formatDuration(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) {
        return null;
    }
    const totalSeconds = Math.max(0, Math.ceil(seconds));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const remSeconds = totalSeconds % 60;
    const parts = [];
    if (hours) {
        parts.push(`${hours}h`);
    }
    if (minutes) {
        parts.push(`${minutes}m`);
    }
    if (!hours && (parts.length === 0 || parts.length < 2)) {
        parts.push(`${remSeconds}s`);
    }
    if (parts.length === 0) {
        parts.push('0s');
    }
    return parts.join(' ');
}

function formatOptions(options) {
    if (!options || typeof options !== 'object') {
        return { label: '-', tooltip: '' };
    }
    const safe = options;
    const parts = [
        `Smooth ${safe.smooth_window_frames ?? '-'}`,
        `Prom ${formatDecimal(safe.prominence_ratio ?? 0)}`,
        `Merge ${formatDecimal(safe.merge_threshold_ms ?? 0, 0)}ms`,
        `FFT ${(safe.fft_denoise ? 'on' : 'off')}@${safe.fft_frames_per_component ?? '-'}`,
    ];
    const label = parts.join(' · ');
    const tooltip = JSON.stringify(safe, null, 2);
    return { label, tooltip };
}

function isSupportedVideo(path) {
    if (typeof path !== 'string') return false;
    const trimmed = path.trim().toLowerCase();
    if (!trimmed) return false;
    const index = trimmed.lastIndexOf('.');
    if (index === -1) return false;
    return VIDEO_EXTENSIONS.has(trimmed.slice(index));
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
        return pathname.replace(/\\/g, '/');
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
        if (typeof file.fullPath === 'string' && file.fullPath.length) return file.fullPath;
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
        const missing = [];
        files.forEach((file) => {
            if (file?.path) {
                paths.push(file.path);
            } else {
                missing.push(file?.name);
            }
        });
        if (missing.length) {
            toast('browser_no_path', 'Browser security blocks access to file paths.', 'error');
        }
        enqueuePaths(paths);
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

function getSelectedModelPath() {
    ensureElements();
    if (!modelSelect) return undefined;
    const value = modelSelect.value;
    return value ? value : undefined;
}

function isVrModel(model) {
    if (!model) return false;
    const identifier = String(model.display_name || model.name || model.path || '').toLowerCase();
    return identifier.includes('vr');
}

function filterModels(models) {
    if (!Array.isArray(models)) return [];
    return models.filter((model) => {
        const vr = isVrModel(model);
        return modelFilterVrOnly ? vr : !vr;
    });
}

function renderModels(models) {
    ensureElements();
    if (!modelSelect) return;
    modelSelect.innerHTML = '';
    const filtered = filterModels(models || []);
    if (!filtered.length) {
        const option = document.createElement('option');
        option.value = '';
        const emptyKey = modelFilterVrOnly ? 'add.no_models_vr' : 'add.no_models_non_vr';
        option.dataset.i18n = emptyKey;
        option.textContent = I18n.t(emptyKey);
        modelSelect.appendChild(option);
        modelSelect.disabled = true;
        I18n.apply(modelSelect);
        return;
    }
    modelSelect.disabled = false;
    filtered.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.display_name || model.name || model.path;
        modelSelect.appendChild(option);
    });
    const defaultModel = State.get('settings')?.default_model_path;
    const preferred = filtered.some((model) => model.path === defaultModel)
        ? defaultModel
        : filtered[0]?.path;
    if (preferred) {
        modelSelect.value = preferred;
    }
    if (!modelSelect.value && filtered[0]) {
        modelSelect.selectedIndex = 0;
    }
    I18n.apply(modelSelect);
}

function attachModelFilter() {
    ensureElements();
    if (!modelFilterToggle) return;
    modelFilterToggle.checked = modelFilterVrOnly;
    if (modelFilterToggle.dataset.filterBound) return;
    modelFilterToggle.addEventListener('change', () => {
        modelFilterVrOnly = Boolean(modelFilterToggle.checked);
        renderModels(State.get('models') || []);
    });
    modelFilterToggle.dataset.filterBound = '1';
}

function setDropzoneBusy(busy) {
    ensureElements();
    if (!dropZone) return;
    dropZone.classList.toggle('dropzone--busy', busy);
}

function cleanPaths(paths) {
    if (!Array.isArray(paths)) return { accepted: [], rejected: 0 };
    const accepted = [];
    let rejected = 0;
    const seen = new Set();
    paths.forEach((raw) => {
        if (typeof raw !== 'string') return;
        const trimmed = raw.trim();
        if (!trimmed) return;
        const key = trimmed.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        if (!isSupportedVideo(trimmed)) {
            rejected += 1;
            return;
        }
        accepted.push(trimmed);
    });
    return { accepted, rejected };
}

async function enqueuePaths(paths, options = {}) {
    const { accepted, rejected } = cleanPaths(paths);
    if (!accepted.length) {
        if (rejected > 0) {
            toast('unsupported', 'Unsupported file type.', 'error');
        } else {
            toast('drop_skipped', 'No new files were queued.', 'info');
        }
        return;
    }
    if (enqueueBusy) {
        toast('drop_failed', 'Queue is busy, try again shortly.', 'info');
        return;
    }
    enqueueBusy = true;
    setDropzoneBusy(true);
    try {
        const payload = {
            video_paths: accepted,
            model_path: getSelectedModelPath(),
            postprocess_options: null,
            recursive: Boolean(options?.recursive),
        };
        const result = await Api.post('/api/queue/add', payload);
        if (result?.added_count) {
            toast('added', 'Added to queue.', 'success', result.added_count);
        } else if (Array.isArray(result?.skipped) && result.skipped.length) {
            toast('skipped', 'All files skipped.', 'info');
        } else {
            toast('skipped', 'No files added.', 'info');
        }
        await refreshQueue();
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        enqueueBusy = false;
        setDropzoneBusy(false);
    }
}

function attachDropZone() {
    ensureElements();
    if (!dropZone || dropZone.dataset.dropBound) return;
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
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('drag-over');
        const paths = pathsFromDrop(event);
        if (!paths.length) {
            toast('drop_skipped', 'No files detected.', 'info');
            return;
        }
        enqueuePaths(paths);
    });
    dropZone.dataset.dropBound = '1';
}

function attachFilePickers() {
    ensureElements();
    if (browseFilesButton && !browseFilesButton.dataset.bound) {
        browseFilesButton.addEventListener('click', async () => {
            const files = await Native.selectFiles();
            if (Array.isArray(files) && files.length) {
                enqueuePaths(files);
                return;
            }
            if (!Native.hasNativeDialogs()) {
                openHtmlFilePicker();
                return;
            }
            toast('dialog', 'File dialog not available.', 'error');
        });
        browseFilesButton.dataset.bound = '1';
    }
}

function attachManualPathInput() {
    ensureElements();
    if (manualPathButton && !manualPathButton.dataset.bound) {
        manualPathButton.addEventListener('click', (event) => {
            event.preventDefault();
            if (!manualPathInput) return;
            const raw = manualPathInput.value.trim();
            if (!raw) {
                showToast(I18n.t('queue.manual_empty') || 'Enter a video path first.', 'info');
                manualPathInput.focus();
                return;
            }
            enqueuePaths([raw]);
            manualPathInput.value = '';
            manualPathInput.focus();
        });
        manualPathButton.dataset.bound = '1';
    }
    if (manualPathInput && !manualPathInput.dataset.bound) {
        manualPathInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                manualPathButton?.click();
            }
        });
        manualPathInput.dataset.bound = '1';
    }
}

function resolveStatusLabel(status) {
    const normalized = String(status || '').toLowerCase();
    const key = STATUS_LABELS[normalized];
    if (key) {
        const translated = I18n.t(key);
        if (translated && translated !== key) {
            return translated;
        }
    }
    return normalized || '-';
}

function buildStatsCell(job) {
    const cell = document.createElement('td');
    cell.className = 'queue-stats';
    const statusLine = document.createElement('div');
    statusLine.className = 'queue-stats__status';
    const statusLabel = resolveStatusLabel(job?.status);
    statusLine.textContent = `${statusLabel} (${formatProgress(job.progress)})`;
    if (job?.message) {
        statusLine.setAttribute('title', job.message);
    }

    const countsLine = document.createElement('div');
    countsLine.className = 'queue-stats__counts text-muted';
    const total = Number.isFinite(job?.frames_total) && job.frames_total > 0
        ? Number(job.frames_total)
        : Number.isFinite(job?.frame_count) && job.frame_count > 0
            ? Number(job.frame_count)
            : null;
    const preprocessed = Number(job?.frames_preprocessed || 0);
    const inferred = Number(job?.frames_inferred || 0);
    if (total) {
        countsLine.textContent = `${I18n.t('queue.stats.pre')} ${preprocessed}/${total} · ${I18n.t('queue.stats.infer')} ${inferred}/${total}`;
    } else {
        countsLine.textContent = `${I18n.t('queue.stats.pre')} ${preprocessed} · ${I18n.t('queue.stats.infer')} ${inferred}`;
    }

    cell.appendChild(statusLine);
    cell.appendChild(countsLine);
    const statusValue = String(job?.status || '').toLowerCase();
    const etaSeconds = Number(job?.eta_seconds);
    const elapsedSeconds = Number(job?.elapsed_seconds);
    const timingParts = [];
    if (statusValue === 'processing' && Number.isFinite(etaSeconds) && etaSeconds >= 0) {
        const formattedEta = formatDuration(etaSeconds);
        if (formattedEta) {
            timingParts.push(`${I18n.t('queue.stats.eta')} ${formattedEta}`);
        }
    }
    if (Number.isFinite(elapsedSeconds) && elapsedSeconds > 0 && statusValue !== 'pending') {
        const formattedElapsed = formatDuration(elapsedSeconds);
        if (formattedElapsed) {
            timingParts.push(`${I18n.t('queue.stats.elapsed')} ${formattedElapsed}`);
        }
    }
    if (timingParts.length > 0) {
        const timingLine = document.createElement('div');
        timingLine.className = 'queue-stats__timing text-muted';
        timingLine.textContent = timingParts.join(' · ');
        cell.appendChild(timingLine);
    }
    return cell;
}

function isCancellable(status) {
    return CANCELLABLE_STATUSES.has(String(status).toLowerCase());
}

function canOpenViewer(job) {
    return String(job.status).toLowerCase() === 'completed' && Boolean(job.prediction_path);
}

function openViewer(jobId) {
    if (!jobId) return;
    document.dispatchEvent(new CustomEvent('app:navigate', {
        detail: {
            view: 'viewer',
            payload: { jobId },
        },
    }));
}

function openCsv(job) {
    const target = job.prediction_path;
    if (!target) {
        showToast(I18n.t('queue.no_csv') || 'No prediction file yet.', 'error');
        return;
    }
    const bridge = window.pywebview?.api;
    if (bridge?.open_path) {
        bridge.open_path(target);
        return;
    }
    navigator.clipboard?.writeText(target);
    showToast(I18n.t('common.path_copied'));
}

function deriveDirectory(path) {
    if (!path) return null;
    const normalized = path.replace(/\\/g, '/');
    const parts = normalized.split('/');
    if (parts.length <= 1) {
        return normalized;
    }
    parts.pop();
    const dir = parts.join('/');
    return dir || normalized;
}

function openFolder(job) {
    const targetDir = deriveDirectory(job.video_path || job.prediction_path || job.script_path);
    if (!targetDir) {
        showToast(I18n.t('queue.no_folder') || 'No folder available.', 'error');
        return;
    }
    const bridge = window.pywebview?.api;
    if (bridge?.open_path) {
        bridge.open_path(targetDir);
        return;
    }
    navigator.clipboard?.writeText(targetDir);
    showToast(I18n.t('common.path_copied'));
}

function buildActionCell(job) {
    const cell = document.createElement('td');
    cell.className = 'queue-actions';
    const actions = [];
    if (canOpenViewer(job)) {
        const viewBtn = document.createElement('button');
        viewBtn.className = 'button button--primary';
        viewBtn.dataset.action = 'open-viewer';
        viewBtn.dataset.i18n = 'queue.open_viewer';
        viewBtn.textContent = I18n.t('queue.open_viewer');
        viewBtn.addEventListener('click', () => openViewer(job.id));
        actions.push(viewBtn);
        const csvBtn = document.createElement('button');
        csvBtn.className = 'button button--secondary';
        csvBtn.dataset.action = 'open-csv';
        csvBtn.dataset.i18n = 'queue.open_csv';
        csvBtn.textContent = I18n.t('queue.open_csv');
        csvBtn.addEventListener('click', () => openCsv(job));
        actions.push(csvBtn);
        const folderBtn = document.createElement('button');
        folderBtn.className = 'button button--secondary';
        folderBtn.dataset.action = 'open-folder';
        folderBtn.dataset.i18n = 'queue.open_folder';
        folderBtn.textContent = I18n.t('queue.open_folder');
        folderBtn.addEventListener('click', () => openFolder(job));
        actions.push(folderBtn);
    }
    if (isCancellable(job.status)) {
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'button button--ghost';
        cancelBtn.dataset.action = 'cancel';
        cancelBtn.dataset.i18n = 'queue.cancel';
        cancelBtn.textContent = I18n.t('queue.cancel');
        cancelBtn.addEventListener('click', () => cancelJob(job.id));
        actions.push(cancelBtn);
    }
    if (!actions.length) {
        cell.textContent = '—';
        cell.classList.add('text-muted');
        return cell;
    }
    const wrapper = document.createElement('div');
    wrapper.className = 'queue-actions__group';
    actions.forEach((button) => wrapper.appendChild(button));
    cell.appendChild(wrapper);
    return cell;
}

function renderQueue(queue) {
    ensureElements();
    if (!queueTableBody) return;
    queueTableBody.innerHTML = '';
    if (!queue || queue.length === 0) {
        const row = document.createElement('tr');
        row.className = 'text-muted';
        row.innerHTML = '<td colspan="5" data-i18n="queue.empty">Queue is empty.</td>';
        queueTableBody.appendChild(row);
        I18n.apply();
        return;
    }
    queue.forEach((job) => {
        const row = document.createElement('tr');
        const modelName = job.model_path ? job.model_path.split(PATH_SPLIT_REGEX).pop() : '-';
        const videoCell = document.createElement('td');
        videoCell.textContent = job.video_name || job.video_path;
        const statsCell = buildStatsCell(job);
        const modelCell = document.createElement('td');
        modelCell.textContent = modelName;
        const optionsCell = document.createElement('td');
        const optionsSpan = document.createElement('span');
        optionsSpan.className = 'queue-options';
        const { label, tooltip } = formatOptions(job.postprocess_options);
        optionsSpan.textContent = label;
        if (tooltip) {
            optionsSpan.setAttribute('title', tooltip);
        }
        optionsCell.appendChild(optionsSpan);
        const actionsCell = buildActionCell(job);
        row.appendChild(videoCell);
        row.appendChild(statsCell);
        row.appendChild(modelCell);
        row.appendChild(optionsCell);
        row.appendChild(actionsCell);
        queueTableBody.appendChild(row);
    });
    I18n.apply();
}

async function refreshQueue() {
    try {
        const data = await Api.get('/api/queue');
        State.setQueue(data || []);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function clearFinished() {
    try {
        await Api.post('/api/queue/clear_finished');
        toast('cleared', 'Finished jobs cleared.');
        await refreshQueue();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function cancelJob(jobId) {
    try {
        await Api.post(`/api/queue/${jobId}/cancel`);
        toast('cancelled', 'Job cancelled.');
        await refreshQueue();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function subscribeState() {
    State.subscribe('queue', renderQueue);
    State.subscribe('models', renderModels);
    State.subscribe('settings', () => {
        renderModels(State.get('models') || []);
    });
}

function startAutoRefresh() {
    if (refreshTimer !== null) return;
    refreshTimer = window.setInterval(refreshQueue, AUTO_REFRESH_MS);
}

function stopAutoRefresh() {
    if (refreshTimer === null) return;
    window.clearInterval(refreshTimer);
    refreshTimer = null;
}

function handleActiveView(view) {
    if (view === 'queue') {
        startAutoRefresh();
    } else {
        stopAutoRefresh();
    }
}

export const QueueView = {
    init() {
        ensureElements();
        refreshButton?.addEventListener('click', refreshQueue);
        clearButton?.addEventListener('click', clearFinished);
        attachModelFilter();
        attachFilePickers();
        attachManualPathInput();
        attachDropZone();
        renderQueue(State.get('queue') || []);
        renderModels(State.get('models') || []);
        subscribeState();
        State.subscribe('activeView', handleActiveView);
        if (State.get('activeView') === 'queue') {
            startAutoRefresh();
        }
    },
    show() {
        renderModels(State.get('models') || []);
        refreshQueue();
        startAutoRefresh();
    },
};

export default QueueView;

if (typeof window !== 'undefined') {
    window.addEventListener('native:files-dropped', (event) => {
        const detail = event?.detail;
        const paths = Array.isArray(detail) ? detail : [];
        if (!paths.length) {
            toast('drop_skipped', 'No files detected.', 'info');
            return;
        }
        enqueuePaths(paths);
    });
}
