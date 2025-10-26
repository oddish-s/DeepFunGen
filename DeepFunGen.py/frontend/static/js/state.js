const subscribers = new Map();

const state = {
    theme: localStorage.getItem('app.theme') || 'dark',
    language: localStorage.getItem('app.language') || 'en',
    models: [],
    queue: [],
    settings: null,
    activeView: 'add',
};

function notify(key, value) {
    if (!subscribers.has(key)) {
        return;
    }
    for (const handler of subscribers.get(key)) {
        try {
            handler(value);
        } catch (error) {
            console.error('state subscriber error', error);
        }
    }
}

export const State = {
    get(key) {
        return state[key];
    },
    setTheme(theme) {
        if (!theme) return;
        state.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('app.theme', theme);
        notify('theme', theme);
    },
    setLanguage(language) {
        if (!language) return;
        state.language = language;
        localStorage.setItem('app.language', language);
        notify('language', language);
    },
    setModels(models) {
        state.models = Array.isArray(models) ? models : [];
        notify('models', state.models);
    },
    setQueue(items) {
        state.queue = Array.isArray(items) ? items : [];
        notify('queue', state.queue);
    },
    setSettings(settings) {
        state.settings = settings || null;
        notify('settings', state.settings);
    },
    setActiveView(view) {
        if (!view) return;
        state.activeView = view;
        notify('activeView', view);
    },
    subscribe(key, handler) {
        if (!subscribers.has(key)) {
            subscribers.set(key, new Set());
        }
        subscribers.get(key).add(handler);
        return () => {
            subscribers.get(key)?.delete(handler);
        };
    },
};

// Initialise theme immediately
State.setTheme(state.theme);

export default State;
