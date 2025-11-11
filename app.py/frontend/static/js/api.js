const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function request(path, options = {}) {
    const response = await fetch(path, options);
    let data = null;
    try {
        data = await response.json();
    } catch (err) {
        // no body
    }
    if (!response.ok) {
        const message = data?.detail || response.statusText || 'Request failed';
        throw new Error(message);
    }
    return data;
}

export const Api = {
    get(path) {
        return request(path, { method: 'GET' });
    },
    post(path, payload) {
        return request(path, {
            method: 'POST',
            headers: JSON_HEADERS,
            body: JSON.stringify(payload ?? {}),
        });
    },
    stream(path, { onMessage, onError } = {}) {
        const source = new EventSource(path);
        if (typeof onMessage === 'function') {
            source.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage(data);
                } catch (err) {
                    console.error('Failed to parse SSE payload', err);
                }
            };
        }
        if (typeof onError === 'function') {
            source.onerror = onError;
        }
        return source;
    },
};

export default Api;