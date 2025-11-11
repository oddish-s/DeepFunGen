import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';

const SAVE_DEBOUNCE_MS = 400;

const state = {
    saveTimer: null,
    saving: false,
    pendingSave: false,
    suppressNextSave: false,
    saveStatus: null,
    saveStatusDetail: '',
    saveClearTimer: null,
    currentVersion: null,
    update: {
        kind: null,
        latestVersion: null,
        url: null,
        error: null,
    },
    fetchingUpdate: false,
    fetchedUpdateOnce: false,
};

let saveStatusLabel;
let updateStatusLabel;
let checkUpdatesButton;
let currentVersionLabel;
let appVersionLabel;

function ensureElements() {
    saveStatusLabel = saveStatusLabel || document.getElementById('settings-status');
    updateStatusLabel = updateStatusLabel || document.getElementById('settings-update-status');
    checkUpdatesButton = checkUpdatesButton || document.getElementById('settings-check-updates');
    currentVersionLabel = currentVersionLabel || document.getElementById('settings-current-version');
    appVersionLabel = appVersionLabel || document.getElementById('app-version');
}

function getMessages() {
    const locale = State.get('language') === 'ko' ? 'ko' : 'en';
    return {
        save: locale === 'ko'
            ? { saving: '저장 중...', saved: '저장됨', error: '오류' }
            : { saving: 'Saving...', saved: 'Saved', error: 'Error' },
        update: locale === 'ko'
            ? {
                checking: '업데이트 확인 중...',
                upToDate: (latest) => (latest ? `최신 상태입니다 (최신 ${latest})` : '최신 상태입니다'),
                available: (latest) => (latest ? `새 업데이트: ${latest}` : '새 업데이트가 있습니다'),
                error: (reason) => (reason ? `확인 실패: ${reason}` : '업데이트를 확인하지 못했습니다.'),
                link: '릴리스 확인',
                toastAvailable: (latest) => (latest ? `새 업데이트: ${latest}` : '새 업데이트가 있습니다.'),
                toastUpToDate: '이미 최신 버전입니다.',
            }
            : {
                checking: 'Checking for updates...',
                upToDate: (latest) => (latest ? `Up to date (latest ${latest})` : 'Up to date'),
                available: (latest) => (latest ? `Update available: ${latest}` : 'Update available'),
                error: (reason) => (reason ? `Failed to check: ${reason}` : 'Failed to check for updates.'),
                link: 'View release',
                toastAvailable: (latest) => (latest ? `Update available: ${latest}` : 'Update available.'),
                toastUpToDate: 'You are on the latest version.',
            },
    };
}

function setSaveStatus(kind, detail = '') {
    state.saveStatus = kind;
    state.saveStatusDetail = detail;
    renderSaveStatus();
    if (state.saveClearTimer) {
        window.clearTimeout(state.saveClearTimer);
        state.saveClearTimer = null;
    }
    if (kind === 'saved') {
        state.saveClearTimer = window.setTimeout(() => {
            state.saveStatus = null;
            state.saveStatusDetail = '';
            renderSaveStatus();
        }, 2000);
    }
}

function renderSaveStatus() {
    ensureElements();
    if (!saveStatusLabel) {
        return;
    }
    const messages = getMessages().save;
    let text = '';
    if (state.saveStatus === 'saving') {
        text = messages.saving;
    } else if (state.saveStatus === 'saved') {
        text = messages.saved;
    } else if (state.saveStatus === 'error') {
        text = `${messages.error}: ${state.saveStatusDetail}`;
    }
    saveStatusLabel.textContent = text;
}

function setUpdateState(kind, payload = {}) {
    state.update = {
        kind,
        latestVersion: payload.latestVersion || null,
        url: payload.url || null,
        error: payload.error || null,
    };
    renderUpdateStatus();
}

function renderUpdateStatus() {
    ensureElements();
    if (!updateStatusLabel) {
        return;
    }
    const { kind, latestVersion, url, error } = state.update;
    const messages = getMessages().update;
    updateStatusLabel.textContent = '';
    updateStatusLabel.innerHTML = '';
    if (!kind) {
        return;
    }
    if (kind === 'checking') {
        updateStatusLabel.textContent = messages.checking;
        return;
    }
    if (kind === 'up_to_date') {
        updateStatusLabel.textContent = messages.upToDate(latestVersion);
        return;
    }
    if (kind === 'available') {
        const message = document.createElement('span');
        message.textContent = messages.available(latestVersion);
        updateStatusLabel.appendChild(message);

        const button = document.createElement('a');
        button.className = 'button button--secondary settings__update-button';
        button.href = url || 'https://github.com/oddish-s/DeepFunGen/releases';
        button.target = '_blank';
        button.rel = 'noopener noreferrer';
        button.textContent = messages.link;
        updateStatusLabel.appendChild(button);
        return;
    }
    if (kind === 'error') {
        updateStatusLabel.textContent = messages.error(error || '');
    }
}

function updateCurrentVersion(value) {
    ensureElements();
    if (!value) {
        return;
    }
    const text = String(value).trim();
    if (!text) {
        return;
    }
    state.currentVersion = text;
    if (currentVersionLabel) {
        currentVersionLabel.textContent = text;
    }
}

function scheduleSave() {
    ensureElements();
    if (state.suppressNextSave) {
        return;
    }
    if (state.saveTimer) {
        window.clearTimeout(state.saveTimer);
    }
    setSaveStatus('saving');
    state.saveTimer = window.setTimeout(() => {
        state.saveTimer = null;
        saveSettings();
    }, SAVE_DEBOUNCE_MS);
}

function formatError(error) {
    if (!error) {
        return '';
    }
    if (typeof error === 'string') {
        return error;
    }
    if (error.message) {
        return error.message;
    }
    return String(error);
}

async function saveSettings() {
    if (state.saving) {
        state.pendingSave = true;
        return;
    }
    state.saving = true;
    const current = State.get('settings') || {};
    const payload = {
        theme: State.get('theme'),
        language: State.get('language'),
        default_model_path: current.default_model_path ?? null,
        default_postprocess: current.default_postprocess ?? null,
    };
    try {
        const result = await Api.post('/api/settings', payload);
        state.suppressNextSave = true;
        const merged = state.currentVersion
            ? { ...result, current_version: state.currentVersion }
            : result;
        State.setSettings(merged);
        window.setTimeout(() => {
            state.suppressNextSave = false;
        }, 0);
        setSaveStatus('saved');
    } catch (error) {
        const message = formatError(error) || 'Failed to save settings.';
        showToast(message, 'error');
        setSaveStatus('error', message);
    } finally {
        state.saving = false;
        if (state.pendingSave) {
            state.pendingSave = false;
            scheduleSave();
        }
    }
}

async function checkForUpdates(manual = false) {
    ensureElements();
    if (state.fetchingUpdate) {
        return;
    }
    state.fetchingUpdate = true;
    setUpdateState('checking');

    const messages = getMessages().update;

    try {
        const data = await Api.get('/api/system/update');
        if (data?.current_version) {
            updateCurrentVersion(data.current_version);
        }
        if (data?.error) {
            setUpdateState('error', { error: data.error });
            if (manual) {
                showToast(messages.error(data.error), 'error');
            }
            return;
        }
        if (data?.has_update) {
            setUpdateState('available', { latestVersion: data.latest_version, url: data.latest_url });
            if (manual) {
                showToast(messages.toastAvailable(data.latest_version), 'info');
            }
            return;
        }
        if (data?.up_to_date) {
            setUpdateState('up_to_date', { latestVersion: data.latest_version });
            if (manual) {
                showToast(messages.toastUpToDate, 'success');
            }
            return;
        }
        if (data?.latest_version) {
            setUpdateState('up_to_date', { latestVersion: data.latest_version });
        } else {
            setUpdateState(null);
        }
    } catch (error) {
        const message = formatError(error);
        setUpdateState('error', { error: message });
        if (manual) {
            showToast(messages.error(message), 'error');
        }
    } finally {
        state.fetchingUpdate = false;
        state.fetchedUpdateOnce = true;
    }
}

function renderSettings(settings) {
    ensureElements();
    document.querySelectorAll('[data-theme-value]').forEach((btn) => {
        const value = btn.getAttribute('data-theme-value');
        btn.classList.toggle('button--secondary', State.get('theme') !== value);
    });
    document.querySelectorAll('[data-lang-value]').forEach((btn) => {
        const value = btn.getAttribute('data-lang-value');
        btn.classList.toggle('button--secondary', State.get('language') !== value);
    });
    if (settings?.current_version) {
        updateCurrentVersion(settings.current_version);
    }
}

export const SettingsView = {
    init() {
        ensureElements();
        document.querySelectorAll('[data-theme-value]').forEach((btn) => {
            btn.addEventListener('click', () => State.setTheme(btn.getAttribute('data-theme-value')));
        });
        document.querySelectorAll('[data-lang-value]').forEach((btn) => {
            btn.addEventListener('click', () => State.setLanguage(btn.getAttribute('data-lang-value')));
        });
        checkUpdatesButton?.addEventListener('click', () => checkForUpdates(true));

        State.subscribe('metadata', (info) => {
            if (info?.version) {
                updateCurrentVersion(info.version);
            }
        });
        State.subscribe('settings', (settings) => {
            state.suppressNextSave = true;
            renderSettings(settings);
            renderSaveStatus();
            renderUpdateStatus();
            window.setTimeout(() => {
                state.suppressNextSave = false;
            }, 0);
        });
        State.subscribe('theme', () => {
            renderSettings(State.get('settings'));
            renderSaveStatus();
            if (!state.suppressNextSave) {
                scheduleSave();
            }
        });
        State.subscribe('language', () => {
            renderSettings(State.get('settings'));
            renderSaveStatus();
            renderUpdateStatus();
            if (!state.suppressNextSave) {
                scheduleSave();
            }
        });

        renderSettings(State.get('settings'));
        renderSaveStatus();
        renderUpdateStatus();
        const metadata = State.get('metadata');
        if (metadata?.version) {
            updateCurrentVersion(metadata.version);
        }
    },
    show() {
        renderSettings(State.get('settings'));
        renderSaveStatus();
        renderUpdateStatus();
        if (!state.fetchedUpdateOnce) {
            checkForUpdates(false);
        }
    },
};

export default SettingsView;
