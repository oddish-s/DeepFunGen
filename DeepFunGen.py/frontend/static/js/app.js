import Api from './api.js';
import { State } from './state.js';
import { AddView } from './views/add.js';
import { QueueView } from './views/queue.js';
import { ViewerView } from './views/viewer.js';
import { LogsView } from './views/logs.js';
import { SettingsView } from './views/settings.js';
import I18n from './i18n.js';

const views = {
    add: AddView,
    queue: QueueView,
    viewer: ViewerView,
    logs: LogsView,
    settings: SettingsView,
};

const navItems = [];
let statusTimer = null;

function initNavigation() {
    const container = document.getElementById('nav-items');
    if (!container) return;
    container.querySelectorAll('.nav__item').forEach((item) => {
        const view = item.getAttribute('data-view');
        if (!view) return;
        navItems.push(item);
        item.addEventListener('click', () => navigate(view));
    });
}

function navigate(view, payload = {}) {
    if (!views[view]) return;
    const hasNavTarget = navItems.some((item) => item.getAttribute('data-view') === view);
    if (hasNavTarget) {
        navItems.forEach((item) => {
            const isActive = item.getAttribute('data-view') === view;
            item.classList.toggle('nav__item--active', isActive);
        });
    }
    document.querySelectorAll('.view').forEach((section) => {
        const isActive = section.id === `view-${view}`;
        section.classList.toggle('view--active', isActive);
    });
    State.setActiveView(view);
    I18n.apply();
    if (typeof views[view].show === 'function') {
        views[view].show(payload);
    }
}

async function refreshModels() {
    try {
        const data = await Api.get('/api/models');
        State.setModels(data?.models || []);
        if (data?.execution_provider) {
            const provider = document.getElementById('status-provider');
            if (provider) provider.textContent = data.execution_provider;
        }
    } catch (error) {
        console.warn('Failed to fetch models', error);
    }
}

async function refreshQueue() {
    try {
        const data = await Api.get('/api/queue');
        State.setQueue(data || []);
    } catch (error) {
        console.warn('Failed to fetch queue', error);
    }
}

async function refreshSettings() {
    try {
        const data = await Api.get('/api/settings');
        State.setSettings(data);
    } catch (error) {
        console.warn('Failed to fetch settings', error);
    }
}

async function refreshStatusBar() {
    try {
        const data = await Api.get('/api/system/status');
        const gpuBlock = document.getElementById('status-gpu-block');
        const gpuValue = document.getElementById('status-gpu');
        if (gpuBlock && gpuValue) {
            if (data.gpu_usage === null || data.gpu_usage === undefined) {
                gpuBlock.classList.add('hidden');
            } else {
                gpuBlock.classList.remove('hidden');
                const gpuPercent = Math.max(0, Math.round(Number(data.gpu_usage) || 0));
                gpuValue.textContent = `${gpuPercent}%`;
            }
        }
        const cpuValue = document.getElementById('status-cpu');
        if (cpuValue) {
            const cpuPercent = Math.max(0, Math.round(Number(data.cpu_usage) || 0));
            cpuValue.textContent = `${cpuPercent}%`;
        }
        document.getElementById('status-queue').textContent = data.queue_total ?? 0;
        const provider = document.getElementById('status-provider');
        if (provider) provider.textContent = data.execution_provider || '-';
        updateBadges();
    } catch (error) {
        console.warn('Failed to update status bar', error);
    }
}

function updateBadges() {
    const queueCount = State.get('queue')?.length || 0;
    const queueBadge = document.querySelector('[data-counter="queue"]');
    if (queueBadge) {
        queueBadge.textContent = queueCount;
        queueBadge.classList.toggle('hidden', queueCount === 0);
    }
}

function initThemeControls() {
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const next = State.get('theme') === 'dark' ? 'light' : 'dark';
            State.setTheme(next);
        });
    }
}

function initLanguageControls() {
    const select = document.getElementById('language-select');
    if (select) {
        select.value = State.get('language');
        select.addEventListener('change', (event) => {
            State.setLanguage(event.target.value);
        });
    }
}

function initViews() {
    Object.values(views).forEach((view) => {
        if (typeof view.init === 'function') {
            view.init();
        }
    });
}

function startTimers() {
    if (statusTimer) window.clearInterval(statusTimer);
    statusTimer = window.setInterval(refreshStatusBar, 2000);
}

async function bootstrap() {
    initNavigation();
    initThemeControls();
    initLanguageControls();
    initViews();

    State.subscribe('queue', updateBadges);

    State.subscribe('settings', (settings) => {
        if (settings?.theme && settings.theme !== State.get('theme')) {
            State.setTheme(settings.theme);
        }
        if (settings?.language && settings.language !== State.get('language')) {
            State.setLanguage(settings.language);
        }
    });

    await Promise.all([
        refreshModels(),
        refreshQueue(),
        refreshSettings(),
        refreshStatusBar(),
    ]);

    const initialView = views[State.get('activeView')] ? State.get('activeView') : 'add';
    navigate(initialView);
    startTimers();
}

document.addEventListener('DOMContentLoaded', bootstrap);

window.AppNavigate = navigate;

document.addEventListener('app:navigate', (event) => {
    const detail = event.detail || {};
    if (!detail.view) return;
    navigate(detail.view, detail.payload || {});
});

export function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    window.setTimeout(() => {
        container.removeChild(toast);
    }, 4200);
}

export default { navigate, showToast };


