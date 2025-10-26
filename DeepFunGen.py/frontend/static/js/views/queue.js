import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';
import I18n from '../i18n.js';

const PATH_SPLIT_REGEX = /[\\/]/;
const CANCELLABLE_STATUSES = new Set(['pending', 'processing']);

let queueTableBody;
let refreshButton;
let clearButton;

function ensureElements() {
    if (!queueTableBody) queueTableBody = document.querySelector('#queue-table tbody');
    if (!refreshButton) refreshButton = document.getElementById('btn-refresh-queue');
    if (!clearButton) clearButton = document.getElementById('btn-clear-finished');
}

function toast(key, fallback, type = 'success') {
    let message;
    switch (key) {
        case 'cancelled':
            message = I18n.t('queue.toast_cancelled');
            break;
        case 'cleared':
            message = I18n.t('queue.toast_cleared');
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
        row.innerHTML = '<td colspan="6" data-i18n="queue.empty">Queue is empty.</td>';
        queueTableBody.appendChild(row);
        I18n.apply();
        return;
    }
    queue.forEach((job) => {
        const row = document.createElement('tr');
        const modelName = job.model_path ? job.model_path.split(PATH_SPLIT_REGEX).pop() : '-';
        const videoCell = document.createElement('td');
        videoCell.textContent = job.video_name || job.video_path;
        const statusCell = document.createElement('td');
        statusCell.textContent = job.status;
        if (job.message) {
            statusCell.setAttribute('title', job.message);
        }
        const progressCell = document.createElement('td');
        progressCell.textContent = formatProgress(job.progress);
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
        row.appendChild(statusCell);
        row.appendChild(progressCell);
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
}

export const QueueView = {
    init() {
        ensureElements();
        refreshButton?.addEventListener('click', refreshQueue);
        clearButton?.addEventListener('click', clearFinished);
        renderQueue(State.get('queue') || []);
        subscribeState();
    },
    show() {
        refreshQueue();
    },
};

export default QueueView;
