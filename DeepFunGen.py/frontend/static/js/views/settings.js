import Api from '../api.js';
import { State } from '../state.js';
import { showToast } from '../app.js';

let statusLabel;

function ensureElements() {
    statusLabel = statusLabel || document.getElementById('settings-status');
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
    if (statusLabel) {
        statusLabel.textContent = '';
    }
}

async function saveSettings() {
    ensureElements();
    const current = State.get('settings') || {};
    const payload = {
        theme: State.get('theme'),
        language: State.get('language'),
        default_model_path: current.default_model_path ?? null,
        default_postprocess: current.default_postprocess ?? null,
    };
    try {
        const result = await Api.post('/api/settings', payload);
        State.setSettings(result);
        showToast('Settings saved.');
        if (statusLabel) {
            statusLabel.textContent = 'Saved';
            window.setTimeout(() => {
                if (statusLabel) statusLabel.textContent = '';
            }, 2000);
        }
    } catch (error) {
        showToast(error.message, 'error');
    }
}

export const SettingsView = {
    init() {
        ensureElements();
        document.getElementById('settings-save')?.addEventListener('click', saveSettings);
        document.querySelectorAll('[data-theme-value]').forEach((btn) => {
            btn.addEventListener('click', () => State.setTheme(btn.getAttribute('data-theme-value')));
        });
        document.querySelectorAll('[data-lang-value]').forEach((btn) => {
            btn.addEventListener('click', () => State.setLanguage(btn.getAttribute('data-lang-value')));
        });
        State.subscribe('settings', renderSettings);
        State.subscribe('theme', () => renderSettings(State.get('settings')));
        State.subscribe('language', () => renderSettings(State.get('settings')));
    },
    show() {
        renderSettings(State.get('settings'));
    },
};

export default SettingsView;
