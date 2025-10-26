# Frontend Implementation Plan

This document outlines the structure and behaviour of the new Python-based desktop frontend. The goal is to replicate and modernise the WinForms UX described in `UI_DesignSpec.md` while consuming the FastAPI backend implemented in `app.py/backend`.

## 1. Technology Stack
- **Rendering shell:** `pywebview` desktop host (already set up via `main.py`).
- **Frontend framework:** Vanilla HTML + CSS + ES modules ? no heavy SPA framework to keep bundle simple for Nuitka packaging.
- **Build tooling:** Static files served directly by FastAPI using `StaticFiles`; no bundler initially.
- **Charting:** Plotly.js (matches existing experimentation and meets interactive requirements).

## 2. Directory Layout
```
frontend/
戍式式 static/
弛   戍式式 css/
弛   弛   戍式式 base.css          # global resets, layout utilities
弛   弛   戍式式 theme.light.css   # light palette overrides
弛   弛   戌式式 theme.dark.css    # dark palette overrides
弛   戍式式 js/
弛   弛   戍式式 app.js            # bootstrap + global navigation + status polling
弛   弛   戍式式 api.js            # fetch helpers wrapping backend endpoints
弛   弛   戍式式 i18n.js           # language switcher + translations (ko/en)
弛   弛   戍式式 state.js          # client-side caches (models, queue, settings)
弛   弛   戌式式 views/
弛   弛       戍式式 queue.js      # staged files + active queue view
弛   弛       戍式式 results.js    # completed jobs gallery
弛   弛       戍式式 viewer.js     # Plotly timeline + funscript preview
弛   弛       戍式式 logs.js       # SSE log stream
弛   弛       戌式式 settings.js   # theme/language/default model controls
弛   戌式式 libs/
弛       戌式式 plotly.min.js     # vendor copy pinned to design spec version
戌式式 templates/
    戍式式 base.html             # chrome (sidebar, top bar, status bar)
    戌式式 views/
        戍式式 queue.html        # upload/staging form partials
        戍式式 results.html      # card list template fragments
        戍式式 viewer.html       # viewer scaffolding (plot container, detail panels)
        戍式式 logs.html         # log console markup
        戌式式 settings.html     # language/theme/model/postprocess panels
```

FastAPI will mount `frontend/static` at `/static` and render `templates/base.html` for `/`.

## 3. Navigation Structure
- **Sidebar (Navigation):** Queue, Results, Viewer, Logs, Settings.
- Selection swaps the `main-view` container. We＊ll mirror WinForms categories: ※Queue§ merges file staging and active queue, ※Results§ lists completed jobs, ※Viewer§ loads a chosen job, ※Logs§ streams backend messages, ※Settings§ handles preferences.
- Provide visual state cues (active highlight, job counts badges for Queue/Results).

## 4. View Behaviour & API Contracts
### 4.1 Queue View
- **Sections:** Drag/drop area, staging list with per-item options, queue table mirroring backend status.
- **API usage:**
  - `GET /api/models` to populate model dropdown and show provider.
  - `POST /api/queue/add` to enqueue selected videos (payload matches `AddJobsRequest`).
  - `GET /api/queue` for queue refresh every 5s.
  - `POST /api/queue/{job_id}/cancel`, `POST /api/queue/clear_finished` for actions.
- **Client state:** keep staged files (not yet submitted) separate from backend queue list. Persist last-used postprocess options in `localStorage` to mirror WinForms convenience.
- **UX:** highlight duplicates/unsupported types before request; show toast or inline status using shared notification component.

### 4.2 Results View
- `GET /api/results` to fetch completed jobs.
- Filtering by filename/model with client-side search; future hook for backend filters.
- Each card shows thumbnail placeholder (to be replaced later with generated image if available), model, completed timestamp, action buttons:
  - ※Open Viewer§ triggers navigation to Viewer view with selected job id.
  - ※Open Folder§ uses `pywebview.api.select_folder`? For now, call new backend future endpoint or show path with copy button (Nuitka packaging constraint). TBD: confirm with user whether we can open OS explorer via backend.

### 4.3 Viewer View
- Loads job detail via `GET /api/jobs/{job_id}` and reads CSV JSON by hitting new endpoint (to be added) or reuse existing result blob. Plan: extend backend with `/api/results/{job_id}` returning processed arrays once pipeline persists details (future work).
- Display components:
  - Plotly chart with processed value vs timestamp.
  - Secondary trace for predicted change if useful.
  - Details panel: job metadata, current cursor timestamp/value, funscript path, download button.
  - Timeline markers from `phase_marker` once backend exposes them.
- Controls: zoom, play/pause scrubbing (basic implementation using Plotly animation frame update).

### 4.4 Logs View
- SSE connection to `/api/logs/stream` using `EventSource`.
- Filters (Info/Warning/Error) rely on message prefix heuristics for now; backend currently sets level to INFO so we may extend later.
- Auto-scroll toggle to pause/resume following.

### 4.5 Settings View
- Fetch current settings via `GET /api/settings`, allow theme/language toggles, default model selection, default postprocess options (subset of `PostprocessOptionsModel`).
- Submit via `POST /api/settings` and update local theme/translation immediately.
- Provide import/export for settings JSON (optional stretch).

### 4.6 Status Bar
- Poll `/api/system/status` every 2s to update GPU/CPU/queue stats and show active execution provider.
- Display queue counts with tooltips for breakdown (pending, processing, completed).

## 5. Internationalisation
- Maintain translation dictionary similar to Claude attempt but cleaned: `i18n.js` with `ko` and `en` keys and fallback to keys when missing.
- UI defaults to Korean per requirements; store selection in `localStorage` and sync with backend `SettingsModel.language`.

## 6. Theming
- CSS variables for colours; body gets `data-theme="dark|light"`. Settings view toggles theme and persists to backend + `localStorage`.
- Provide high-level layout variables (spacing, radius) to satisfy spec (12?16px padding, 8px radius).

## 7. Notifications & Error Handling
- Implement simple notification banner component (top-right) for success/error messages triggered by API helpers.
- All API helpers return structured `{success, data, error}`; errors show localized message.

## 8. Accessibility & Keyboard Support
- Ensure drag/drop area also accepts `Enter`/`Space` to open file dialog.
- Keyboard shortcuts: `Ctrl+V` to paste file path into staging (requires hooking `paste` event and contacting backend to validate path).
- In viewer, add arrow key navigation for frame stepping, space for play/pause.

## 9. Testing Strategy
- Smoke-test JS modules by instantiating views with mock fetch responses (using simple Jest-like runner is overkill; we＊ll rely on manual testing + potential Playwright scripts post MVP).
- Backend integration: use existing API endpoints; add unit tests later for SSE + queue interactions.

## 10. Implementation Order
1. Scaffold templates and base CSS/theme toggles.
2. Implement `api.js`, `state.js`, `app.js` navigation + status bar.
3. Build Queue view (staging, queue table, add/cancel/clear flows).
4. Build Logs view (SSE) ? easier to validate early.
5. Results view cards with navigation to Viewer.
6. Viewer view (requires additional backend endpoint exposing processed data; will implement when wiring view).
7. Settings view + i18n integration.
8. Polish notifications, keyboard shortcuts, and accessibility touches.

## 11. Outstanding Backend Hooks
- Expose processed predictions via `/api/results/{job_id}` (currently planned) for Viewer plot.
- Provide download endpoints for CSV/Funscript (or rely on file path until we add zipped deliverables).
- Optionally add IPC call to open system folder (requires pywebview bridge method).

This plan keeps parity with the WinForms UX while embracing web-native components, and leaves room for incremental enhancements once core flows are stable.