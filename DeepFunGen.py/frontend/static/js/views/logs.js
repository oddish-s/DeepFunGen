import Api from '../api.js';

let output;
let source = null;
let autoScroll = true;
const LOG_LIMIT = 500;

function ensureElements() {
    output = output || document.getElementById('logs-output');
}

function appendLog(entry) {
    ensureElements();
    if (!output) return;
    const time = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    const job = entry.job_name ? `[${entry.job_name}]` : '';
    const level = entry.level ? entry.level.toUpperCase() : 'INFO';
    const line = document.createElement('div');
    line.className = `log-line log-line--${level.toLowerCase()}`;
    line.textContent = `[${time}] [${level}] ${job} ${entry.message}`.trim();
    output.appendChild(line);
    while (output.childNodes.length > LOG_LIMIT) {
        output.removeChild(output.firstChild);
    }
    if (autoScroll) {
        output.scrollTop = output.scrollHeight;
    }
}

function subscribe() {
    if (source) source.close();
    source = Api.stream('/api/logs/stream', {
        onMessage: appendLog,
        onError: () => {
            setTimeout(subscribe, 2000);
        },
    });
}

export const LogsView = {
    init() {
        ensureElements();
        subscribe();
    },
    show() {
        // nothing yet
    },
};

export default LogsView;
