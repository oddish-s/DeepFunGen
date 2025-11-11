import { State } from '../state.js';
import I18n from '../i18n.js';

let container;

function ensureContainer() {
    if (!container) {
        container = document.getElementById('license-content');
    }
    return container;
}

function render(info) {
    const target = ensureContainer();
    if (!target) return;
    target.innerHTML = '';
    const items = info && Array.isArray(info.third_party) ? info.third_party : [];
    if (!items.length) {
        const empty = document.createElement('p');
        empty.className = 'text-muted';
        empty.setAttribute('data-i18n', 'license.none');
        target.appendChild(empty);
        I18n.apply();
        return;
    }
    const list = document.createElement('ul');
    items.forEach((item) => {
        const entry = document.createElement('li');
        entry.className = 'mt-sm';
        const wrapper = document.createElement('div');
        wrapper.className = 'flex flex--col gap-xs';

        const name = document.createElement('strong');
        name.textContent = item.name || '';
        wrapper.appendChild(name);

        if (item.license) {
            const licenseLine = document.createElement('span');
            licenseLine.className = 'text-muted small';
            const label = document.createElement('span');
            label.setAttribute('data-i18n', 'license.license_label');
            label.textContent = 'License';
            licenseLine.appendChild(label);
            licenseLine.appendChild(document.createTextNode(`: ${item.license}`));
            wrapper.appendChild(licenseLine);
        }

        if (item.notes) {
            const notes = document.createElement('span');
            notes.className = 'text-muted small';
            notes.textContent = item.notes;
            wrapper.appendChild(notes);
        }

        if (item.url) {
            const link = document.createElement('a');
            link.href = item.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = item.url;
            wrapper.appendChild(link);
        }

        entry.appendChild(wrapper);
        list.appendChild(entry);
    });
    target.appendChild(list);
    I18n.apply();
}

export const LicenseView = {
    init() {
        ensureContainer();
        State.subscribe('metadata', render);
    },
    show() {
        render(State.get('metadata'));
    },
};

export default LicenseView;
