const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.mpg', '.mpeg', '.wmv'];

function mapExtensions(values) {
    return values.map((ext) => ext.replace('.', '')).filter(Boolean);
}

class NativeBridge {
    constructor() {
        this._bridge = null;
        this._ready = false;
        this._resolveReady = () => {};
        this._readyPromise = new Promise((resolve) => {
            this._resolveReady = resolve;
        });

        const assignBridge = () => {
            if (this._ready) {
                return true;
            }
            if (window.pywebview && window.pywebview.api) {
                this._bridge = window.pywebview.api;
                this._ready = true;
                this._resolveReady(this._bridge);
                return true;
            }
            return false;
        };

        if (!assignBridge()) {
            window.addEventListener('pywebviewready', assignBridge, { once: true });
            const interval = window.setInterval(() => {
                if (assignBridge()) {
                    window.clearInterval(interval);
                }
            }, 400);
            window.setTimeout(() => {
                if (!this._ready) {
                    this._resolveReady(null);
                    window.clearInterval(interval);
                }
            }, 4000);
        }
    }

    async bridge() {
        if (this._ready) {
            return this._bridge;
        }
        return this._readyPromise;
    }

    async selectFiles() {
        const bridge = await this.bridge();
        if (bridge && typeof bridge.select_files === 'function') {
            try {
                const files = await bridge.select_files();
                if (Array.isArray(files)) {
                    return files;
                }
            } catch (error) {
                console.warn('Native select_files failed', error);
            }
        }
        if (window.pywebview && typeof window.pywebview.openFileDialog === 'function') {
            try {
                const result = await window.pywebview.openFileDialog({
                    allowMultiple: true,
                    filters: [
                        { description: 'Video files', extensions: mapExtensions(VIDEO_EXTENSIONS) },
                        { description: 'All files', extensions: ['*'] },
                    ],
                });
                if (Array.isArray(result)) {
                    return result;
                }
            } catch (error) {
                console.warn('openFileDialog failed', error);
            }
        }
        return null;
    }

    async selectFolder() {
        const bridge = await this.bridge();
        if (bridge && typeof bridge.select_folder === 'function') {
            try {
                const folder = await bridge.select_folder();
                if (typeof folder === 'string' && folder.length) {
                    return folder;
                }
            } catch (error) {
                console.warn('Native select_folder failed', error);
            }
        }
        if (window.pywebview && typeof window.pywebview.openDirectoryDialog === 'function') {
            try {
                const result = await window.pywebview.openDirectoryDialog();
                if (Array.isArray(result) && result.length) {
                    return result[0];
                }
            } catch (error) {
                console.warn('openDirectoryDialog failed', error);
            }
        }
        return null;
    }

    hasNativeDialogs() {
        const bridgeReady = Boolean(this._bridge && (this._bridge.select_files || this._bridge.select_folder));
        const fallbackReady = Boolean(window.pywebview && (window.pywebview.openFileDialog || window.pywebview.openDirectoryDialog));
        const pendingBridge = !this._ready && Boolean(window.pywebview);
        return bridgeReady || fallbackReady || pendingBridge;
    }
}

const Native = new NativeBridge();

export default Native;
