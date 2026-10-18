(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    'video-file', 'drop-zone', 'media-workspace', 'player-shell', 'video-player',
    'playback-fallback', 'file-details', 'file-name', 'file-meta', 'source-caption',
    'change-button', 'clear-button', 'analyze-button', 'demo-button', 'demo-notice',
    'global-error', 'global-error-text', 'service-state', 'service-label', 'model-label',
    'availability-note', 'availability-text', 'config-retry', 'upload-formats',
    'progress-block', 'progress-label', 'upload-percent', 'upload-progress', 'progress-detail',
    'result-panel', 'result-empty', 'result-processing', 'result-content', 'result-count',
    'processing-title', 'processing-copy', 'job-reference', 'poll-retry', 'verdict-block',
    'verdict-symbol', 'verdict-eyebrow', 'verdict-text', 'result-meta', 'result-limitations',
    'limitations-list', 'events-heading', 'events-total', 'event-list', 'no-events',
    'download-button', 'timeline-caption', 'timeline-markers', 'timeline-progress',
    'timeline-duration', 'evidence-dialog', 'evidence-image', 'evidence-title',
    'evidence-description', 'seek-evidence-button', 'guide-dialog', 'live-status',
    'privacy-status', 'privacy-title', 'privacy-detail', 'privacy-progress', 'preview-retry', 'processing-eyebrow',
  ].map((id) => [id, $(id)]));

  const state = {
    config: null, configLoading: false, configPromise: null, source: null, file: null,
    busy: false, generation: 0, xhr: null, pollTimer: null, pollAbort: null,
    privacyBusy: false, privacyAbort: null, previewReady: false,
    jobId: null, result: null, duration: 0, events: [], connectionFailures: 0,
    evidenceTime: 0, activeEvent: -1, configRequest: 0,
  };

  function textElement(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function icon(name, className) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#icon-${name}`);
    svg.append(use);
    svg.setAttribute('aria-hidden', 'true');
    if (className) svg.setAttribute('class', className);
    return svg;
  }

  function announce(message) { ui['live-status'].textContent = message; }
  function finiteNumber(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
  function formatTime(seconds) {
    if (!finiteNumber(seconds)) return '—:—';
    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return `${h ? `${h}:` : ''}${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }
  function formatSize(size) {
    return size >= 1024 * 1024 ? `${(size / (1024 * 1024)).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МиБ` : `${Math.max(1, Math.ceil(size / 1024))} КиБ`;
  }
  function eventCount(count) {
    const suffix = count % 100 >= 11 && count % 100 <= 14 ? 'событий' : count % 10 === 1 ? 'событие' : count % 10 >= 2 && count % 10 <= 4 ? 'события' : 'событий';
    return `${count} ${suffix}`;
  }

  function showError(message) {
    ui['global-error-text'].textContent = message;
    ui['global-error'].hidden = false;
  }
  function clearError() { ui['global-error'].hidden = true; }

  function updateAnalyzeButton() {
    ui['analyze-button'].disabled = !state.config?.ready || !state.file || state.busy || state.privacyBusy || state.source === 'demo';
    ui['analyze-button'].querySelector('span').textContent = state.privacyBusy ? 'Готовим предпросмотр' : state.busy && state.source !== 'demo' ? 'Проверка выполняется' : 'Начать проверку';
    ui['preview-retry'].disabled = state.busy || state.privacyBusy;
    const isDemo = state.source === 'demo';
    const demoAction = isDemo ? 'Вернуться к проверке' : 'Демо интерфейса';
    ui['demo-button'].replaceChildren(icon('video'), document.createTextNode(demoAction));
    ui['demo-button'].title = demoAction;
    ui['demo-button'].disabled = !isDemo && state.configLoading;
  }

  function loadConfig() {
    if (state.configPromise) return state.configPromise;
    state.configPromise = fetchConfig().finally(() => { state.configPromise = null; });
    return state.configPromise;
  }

  async function fetchConfig() {
    state.configLoading = true;
    updateAnalyzeButton();
    const request = ++state.configRequest;
    ui['config-retry'].hidden = true;
    ui['service-state'].dataset.state = 'loading';
    ui['service-label'].textContent = 'Подключение к сервису';
    try {
      const response = await fetch('/api/ui/config', { credentials: 'same-origin', cache: 'no-store' });
      if (!response.ok) throw new Error('config');
      const config = await response.json();
      if (request !== state.configRequest) return;
      state.config = config;
      const max = finiteNumber(config.max_upload_mb) && config.max_upload_mb > 0 ? config.max_upload_mb : 100;
      ui['upload-formats'].textContent = `MOV или MP4 · до ${max} МиБ`;
      ui['service-state'].dataset.state = config.ready ? 'ready' : 'unavailable';
      ui['service-label'].textContent = config.ready ? 'Модель готова к проверке' : 'Модель не подключена';
      ui['model-label'].textContent = config.model_name ? `Модель: ${config.model_name}` : 'Модель не подключена';
      ui['availability-note'].classList.toggle('ready', Boolean(config.ready));
      ui['availability-text'].textContent = config.ready
        ? 'Модель подключена. Добавьте видео, чтобы начать проверку.'
        : 'Модель ещё не подключена. Видео можно открыть для предпросмотра или посмотреть демо интерфейса.';
      ui['config-retry'].hidden = Boolean(config.ready);
    } catch {
      if (request !== state.configRequest) return;
      state.config = null;
      ui['service-state'].dataset.state = 'unavailable';
      ui['service-label'].textContent = 'Нет соединения с сервисом';
      ui['availability-note'].classList.remove('ready');
      ui['availability-text'].textContent = 'Не удалось подключиться к сервису. Повторите подключение, чтобы подготовить предпросмотр с размытием лиц.';
      ui['config-retry'].hidden = false;
      ui['model-label'].textContent = 'Сервис недоступен';
    } finally {
      state.configLoading = false;
      updateAnalyzeButton();
    }
  }

  function cancelCurrentWork() {
    state.generation += 1;
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    state.pollAbort?.abort();
    state.pollAbort = null;
    state.privacyAbort?.abort();
    state.privacyAbort = null;
    state.privacyBusy = false;
    if (state.xhr) { state.xhr.abort(); state.xhr = null; }
    state.busy = false;
    state.jobId = null;
    state.connectionFailures = 0;
    if (ui['evidence-dialog'].open) ui['evidence-dialog'].close();
  }

  function resetResults() {
    state.result = null;
    state.events = [];
    state.activeEvent = -1;
    ui['result-empty'].hidden = false;
    ui['result-processing'].hidden = true;
    ui['result-content'].hidden = true;
    ui['result-count'].hidden = true;
    ui['download-button'].hidden = true;
    ui['progress-block'].hidden = true;
    ui['result-panel'].setAttribute('aria-busy', 'false');
    ui['event-list'].replaceChildren();
    ui['timeline-markers'].replaceChildren();
    ui['timeline-caption'].textContent = 'Появится после проверки';
    ui['timeline-progress'].style.width = '0%';
    ui['poll-retry'].hidden = true;
  }

  function releasePreview() {
    ui['video-player'].pause();
    ui['video-player'].removeAttribute('src');
    ui['video-player'].load();
    state.previewReady = false;
    state.duration = 0;
    ui['timeline-duration'].textContent = '—:—';
    ui['playback-fallback'].hidden = true;
    ui['player-shell'].hidden = true;
    ui['player-shell'].classList.remove('portrait');
    ui['privacy-status'].hidden = true;
  }

  function clearVideo() {
    cancelCurrentWork();
    releasePreview();
    resetResults();
    clearError();
    state.file = null;
    state.source = null;
    ui['video-file'].value = '';
    ui['drop-zone'].hidden = false;
    ui['player-shell'].hidden = true;
    ui['file-details'].hidden = true;
    ui['change-button'].hidden = true;
    ui['source-caption'].hidden = false;
    ui['demo-notice'].hidden = true;
    updateAnalyzeButton();
  }

  function returnToCheck() {
    clearVideo();
    ui['drop-zone'].focus();
    announce('Демо закрыто. Добавьте видеозапись для проверки.');
  }

  function updateFileMeta() {
    const parts = [];
    if (state.source === 'demo') parts.push('Исходная разметка');
    else if (state.file) parts.push(formatSize(state.file.size), state.file.name.split('.').pop().toUpperCase());
    if (state.duration > 0) parts.push(formatTime(state.duration));
    ui['file-meta'].textContent = parts.join(' · ');
  }

  function showSelectedFile(filename) {
    ui['drop-zone'].hidden = true;
    ui['file-details'].hidden = false;
    ui['change-button'].hidden = false;
    ui['source-caption'].hidden = true;
    ui['file-name'].textContent = filename;
    updateFileMeta();
  }

  function safeMediaURL(raw) {
    if (typeof raw !== 'string') return null;
    try {
      const url = new URL(raw, window.location.origin);
      if (url.origin !== window.location.origin || url.search || url.hash) return null;
      if (url.pathname !== '/api/ui/demo/video' && !/^\/api\/ui\/(?:previews|jobs)\/[A-Za-z0-9_-]+\/video$/.test(url.pathname)) return null;
      return url.pathname;
    } catch { return null; }
  }

  function showPrivacyStatus(mode, detail) {
    ui['privacy-status'].hidden = false;
    ui['privacy-status'].dataset.state = mode;
    ui['privacy-title'].textContent = mode === 'processing' ? 'Размываем лица для предпросмотра' : mode === 'ready' ? 'Предпросмотр готов' : 'Не удалось подготовить предпросмотр';
    ui['privacy-detail'].textContent = detail || (mode === 'processing'
      ? 'Готовим копию видео для просмотра. Это может занять до двух минут.'
      : mode === 'ready' ? 'Обнаруженные лица размыты локально. Детектор может пропускать лица.'
        : 'Повторите подготовку или выберите другое видео. Оригинальная запись не показывается.');
    ui['privacy-progress'].hidden = mode !== 'processing';
    ui['preview-retry'].hidden = mode !== 'error';
  }

  function displayPreview(src, filename) {
    const safeURL = safeMediaURL(src);
    if (!safeURL) throw new Error('preview-url');
    showSelectedFile(filename);
    state.previewReady = true;
    ui['player-shell'].hidden = false;
    ui['playback-fallback'].hidden = true;
    ui['video-player'].src = safeURL;
    ui['video-player'].load();
    showPrivacyStatus('ready');
  }

  async function preparePreview() {
    if (!state.file || state.source !== 'upload' || state.busy) return;
    state.privacyAbort?.abort();
    const controller = new AbortController();
    state.privacyAbort = controller;
    const generation = state.generation;
    const file = state.file;
    state.privacyBusy = true;
    releasePreview();
    clearError();
    showSelectedFile(file.name);
    showPrivacyStatus('processing');
    updateAnalyzeButton();
    announce('Размываем лица для предпросмотра.');
    try {
      if (!state.config) await loadConfig();
      if (generation !== state.generation || controller.signal.aborted) return;
      if (!state.config) throw new Error('preview-connection');
      const form = new FormData();
      form.append('file', file, file.name);
      const response = await fetch('/api/ui/previews', { method: 'POST', body: form, credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
      if (generation !== state.generation || controller.signal.aborted) return;
      if (response.status === 401 || response.status === 403) state.config = null;
      if (!response.ok) throw new Error('preview-failed');
      const data = await response.json();
      if (generation !== state.generation || controller.signal.aborted) return;
      const safeURL = safeMediaURL(data.video_url);
      if (typeof data.preview_id !== 'string' || !safeURL || safeURL !== `/api/ui/previews/${encodeURIComponent(data.preview_id)}/video`) throw new Error('preview-url');
      if (finiteNumber(data.duration_sec)) state.duration = data.duration_sec;
      displayPreview(safeURL, file.name);
      ui['timeline-duration'].textContent = formatTime(state.duration || null);
      if (state.events.length) renderTimeline();
      announce('Предпросмотр готов. Обнаруженные лица размыты локально. Детектор может пропускать лица.');
    } catch (error) {
      if (generation !== state.generation || controller.signal.aborted) return;
      showPrivacyStatus('error', error.message === 'preview-connection'
        ? 'Нет соединения с сервисом. Повторите подготовку после восстановления соединения. Оригинальная запись не показывается.'
        : 'Повторите подготовку или выберите другое видео. Оригинальная запись не показывается; при доступной модели её можно отправить на анализ.');
      announce('Не удалось подготовить предпросмотр. Оригинальная запись не показывается. Можно повторить подготовку.');
    } finally {
      if (state.privacyAbort === controller) {
        state.privacyAbort = null;
        state.privacyBusy = false;
        updateAnalyzeButton();
      }
    }
  }

  function selectFile(file) {
    if (!file) return;
    const maximum = state.config?.max_upload_mb || 100;
    if (!file.size) { showError('Файл пустой. Выберите видеозапись с содержимым.'); return; }
    if (file.size > maximum * 1024 * 1024) { showError(`Размер файла превышает ${maximum} МиБ. Выберите запись меньшего размера.`); return; }
    const extension = file.name.split('.').pop().toLowerCase();
    if (!['mov', 'mp4'].includes(extension)) {
      showError('Выберите видео в формате MOV или MP4.');
      return;
    }
    cancelCurrentWork();
    releasePreview();
    resetResults();
    clearError();
    state.source = 'upload';
    state.file = file;
    ui['demo-notice'].hidden = true;
    preparePreview();
  }

  function showProcessing(title, copy) {
    ui['result-empty'].hidden = true;
    ui['result-content'].hidden = true;
    ui['result-processing'].hidden = false;
    ui['result-panel'].setAttribute('aria-busy', 'true');
    ui['processing-eyebrow'].textContent = state.source === 'demo' ? 'Пример из исходной разметки' : 'Проверка выполняется';
    ui['processing-title'].textContent = title || 'Анализируем видеозапись';
    ui['processing-copy'].textContent = copy || 'Время зависит от длительности записи и доступности модели. Результат появится здесь.';
    ui['job-reference'].textContent = state.jobId ? `Проверка ${state.jobId}` : '';
  }

  function processingProgress() {
    ui['progress-block'].hidden = false;
    ui['progress-label'].textContent = 'Проверка выполняется';
    ui['upload-percent'].textContent = '';
    ui['upload-progress'].removeAttribute('value');
    ui['upload-progress'].setAttribute('aria-label', 'Проверка выполняется, ожидаем результат');
    ui['progress-detail'].textContent = 'Видео загружено. Ожидаем результат модели.';
  }

  function messageFromResponse(data, fallback) {
    if (typeof data?.error === 'string') return data.error;
    if (typeof data?.detail === 'string') return data.detail;
    if (typeof data?.message === 'string') return data.message;
    return fallback;
  }

  function startAnalysis() {
    if (!state.file || !state.config?.ready || state.busy || state.privacyBusy || state.source === 'demo') return;
    cancelCurrentWork();
    resetResults();
    clearError();
    state.busy = true;
    const generation = state.generation;
    showProcessing('Передаём видеозапись', 'После загрузки сервис начнёт проверку. События и кадры появятся здесь.');
    ui['progress-block'].hidden = false;
    ui['progress-label'].textContent = 'Загрузка видео';
    ui['upload-percent'].textContent = '0%';
    ui['upload-progress'].value = 0;
    ui['upload-progress'].setAttribute('aria-label', 'Прогресс загрузки видео на сервер');
    ui['progress-detail'].textContent = 'Файл передаётся на сервер.';
    updateAnalyzeButton();
    announce('Загружаем видео на сервер.');
    const form = new FormData();
    form.append('file', state.file, state.file.name);
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open('POST', '/api/ui/analyze');
    xhr.withCredentials = true;
    xhr.responseType = 'json';
    xhr.upload.addEventListener('progress', (event) => {
      if (generation !== state.generation || !event.lengthComputable) return;
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      ui['upload-progress'].value = percent;
      ui['upload-percent'].textContent = `${percent}%`;
      ui['progress-detail'].textContent = `${formatSize(event.loaded)} из ${formatSize(event.total)}`;
    });
    xhr.upload.addEventListener('load', () => {
      if (generation !== state.generation) return;
      processingProgress();
      showProcessing();
    });
    xhr.addEventListener('load', () => {
      if (generation !== state.generation) return;
      state.xhr = null;
      const job = xhr.response;
      if (xhr.status < 200 || xhr.status >= 300 || typeof job?.job_id !== 'string') {
        showFailure(messageFromResponse(job, 'Не удалось начать проверку. Повторите попытку.'));
        return;
      }
      state.jobId = job.job_id;
      showProcessing();
      processingProgress();
      announce('Видео загружено. Проверка выполняется.');
      if (job.status === 'done' || job.status === 'failed') handleJob(job, generation);
      else pollJob(state.jobId, generation);
    });
    xhr.addEventListener('error', () => {
      if (generation !== state.generation) return;
      state.xhr = null;
      showFailure('Не удалось передать видео: соединение прервалось. Файл сохранён в окне, можно повторить проверку.');
    });
    xhr.send(form);
  }

  async function pollJob(jobId, generation) {
    if (generation !== state.generation || state.jobId !== jobId || state.pollAbort || !state.busy) return;
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    const controller = new AbortController();
    state.pollAbort = controller;
    let nextDelay = 1800;
    try {
      const response = await fetch(`/api/ui/jobs/${encodeURIComponent(jobId)}`, { credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
      if (generation !== state.generation) return;
      if (response.status === 401 || response.status === 403 || response.status === 404) {
        showFailure(response.status === 404 ? 'Проверка не найдена. Добавьте видео и запустите проверку повторно.' : 'Сессия проверки недоступна. Обновите подключение к сервису и запустите проверку повторно.');
        return;
      }
      if (!response.ok) throw new Error('connection');
      const job = await response.json();
      if (generation !== state.generation) return;
      state.connectionFailures = 0;
      ui['poll-retry'].hidden = true;
      if (job.status === 'done' || job.status === 'failed') { handleJob(job, generation); return; }
      if (job.status !== 'processing') throw new Error('status');
      showProcessing();
      ui['progress-detail'].textContent = 'Видео загружено. Ожидаем результат модели.';
    } catch (error) {
      if (error.name === 'AbortError' || generation !== state.generation) return;
      state.connectionFailures += 1;
      showProcessing('Восстанавливаем соединение', 'Задание сохранено. Попробуем получить результат автоматически; также можно повторить подключение сейчас.');
      ui['poll-retry'].hidden = false;
      ui['progress-detail'].textContent = 'Связь с сервисом прервалась. Повторно загружать видео не нужно.';
      nextDelay = Math.min(20000, 2500 * (2 ** Math.min(state.connectionFailures - 1, 3)));
      if (state.connectionFailures === 1) announce('Связь с сервисом прервалась. Задание сохранено, восстанавливаем соединение.');
    } finally {
      if (state.pollAbort === controller) state.pollAbort = null;
    }
    if (generation === state.generation && state.busy) state.pollTimer = window.setTimeout(() => pollJob(jobId, generation), nextDelay);
  }

  function handleJob(job, generation) {
    if (generation !== state.generation) return;
    if (job.status === 'failed') { showFailure(messageFromResponse(job, 'Сервис не смог обработать видео. Выберите другую запись или повторите проверку.')); return; }
    try { renderResult(job, false); }
    catch { showFailure('Сервис вернул неполный результат проверки. Повторите проверку или обратитесь к администратору сервиса.'); }
  }

  function showFailure(message) {
    state.busy = false;
    ui['progress-block'].hidden = true;
    ui['result-processing'].hidden = true;
    ui['result-content'].hidden = true;
    ui['result-empty'].hidden = false;
    ui['result-panel'].setAttribute('aria-busy', 'false');
    showError(message);
    updateAnalyzeButton();
    announce(`Проверка не завершена. ${message}`);
  }

  function jpegSource(raw) {
    if (typeof raw !== 'string' || raw.length > 16000000) return null;
    const clean = raw.replace(/^data:image\/(?:jpeg|jpg);base64,/i, '').replace(/\s/g, '');
    if (!clean.startsWith('/9j/') || !/^[A-Za-z0-9+/]+={0,2}$/.test(clean)) return null;
    return `data:image/jpeg;base64,${clean}`;
  }

  function safeRuleURL(raw) {
    if (typeof raw !== 'string') return null;
    try {
      const parsed = new URL(raw);
      return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : null;
    } catch { return null; }
  }

  function eventTitle(type) {
    const titles = { unknown: 'Тип события не определён', undetermined: 'Тип события не определён', possible_violation: 'Признаки возможного нарушения', violation: 'Признаки возможного нарушения' };
    return typeof type === 'string' && type.trim() ? (titles[type] || type) : 'Тип события не определён';
  }

  function renderRule(rule) {
    const block = textElement('div', 'rule-block');
    block.append(textElement('span', '', 'Нормативное основание'));
    const title = typeof rule?.title === 'string' ? rule.title.trim() : '';
    const clause = typeof rule?.clause === 'string' ? rule.clause.trim() : '';
    const url = safeRuleURL(rule?.url);
    if (!title && !clause && !url) {
      block.append(textElement('span', '', 'Нормативное основание не определено'));
      return block;
    }
    if (url) {
      const link = textElement('a', '', title || 'Открыть источник');
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      block.append(link);
    } else if (title) block.append(textElement('span', '', title));
    if (clause) block.append(textElement('div', '', clause));
    return block;
  }

  function openEvidence(frame, event, source) {
    const time = finiteNumber(frame.timestamp_sec) ? frame.timestamp_sec : event.timecode_sec;
    state.evidenceTime = finiteNumber(time) ? time : 0;
    ui['evidence-title'].textContent = `Кадр · ${formatTime(state.evidenceTime)}${state.source === 'demo' ? ' · Демо' : ''}`;
    ui['evidence-image'].src = source;
    ui['evidence-image'].alt = `Кадр на ${formatTime(state.evidenceTime)}. ${eventTitle(event.type)}`;
    ui['evidence-description'].textContent = state.source === 'demo' ? 'Кадр из исходного видео. Отметка взята из разметки; модель не запускалась.' : eventTitle(event.type);
    ui['evidence-dialog'].showModal();
  }

  function renderEvent(event, index) {
    const card = textElement('article', 'event-card');
    card.id = `event-${index}`;
    const heading = textElement('div', 'event-header');
    const timeButton = textElement('button', 'timecode');
    timeButton.type = 'button';
    timeButton.append(icon('clock'), document.createTextNode(formatTime(event.timecode_sec)));
    timeButton.setAttribute('aria-label', `Перейти к событию на ${formatTime(event.timecode_sec)}`);
    timeButton.addEventListener('click', () => seekTo(event.timecode_sec, index));
    heading.append(timeButton, textElement('span', 'event-index', `Событие ${String(index + 1).padStart(2, '0')}`));
    card.append(heading, textElement('h4', '', eventTitle(event.type)));
    if (typeof event.description === 'string') card.append(textElement('p', 'event-description', event.description));
    const frames = textElement('div', 'evidence-frames');
    const evidence = Array.isArray(event.evidence_frames) ? event.evidence_frames : Array.isArray(event.evidence) ? event.evidence : [];
    evidence.slice(0, 3).forEach((frame) => {
      if (!frame || typeof frame !== 'object') return;
      const source = jpegSource(frame.image_base64);
      if (!source) return;
      const time = finiteNumber(frame.timestamp_sec) ? frame.timestamp_sec : event.timecode_sec;
      const button = textElement('button', 'frame-button');
      button.type = 'button';
      button.setAttribute('aria-label', `Увеличить кадр на ${formatTime(time)}`);
      const image = document.createElement('img');
      image.src = source;
      image.addEventListener('load', () => button.classList.toggle('portrait', image.naturalHeight > image.naturalWidth));
      image.alt = `Кадр на ${formatTime(time)}`;
      image.loading = 'lazy';
      button.append(image, textElement('span', 'frame-time', formatTime(time)), icon('expand', 'frame-expand'));
      button.addEventListener('click', () => openEvidence(frame, event, source));
      frames.append(button);
    });
    if (frames.childElementCount) card.append(frames);
    card.append(renderRule(event.rule));
    return card;
  }

  function renderResult(job, isDemo) {
    const result = job.result && typeof job.result === 'object' ? { ...job, ...job.result } : job;
    if (!['violation', 'no_violation'].includes(result.verdict) || !Array.isArray(result.violations)) throw new Error('Invalid result');
    state.busy = false;
    state.result = job;
    state.events = result.verdict === 'no_violation' ? [] : result.violations.filter((event) => event && typeof event === 'object');
    if (finiteNumber(result.duration_sec) && result.duration_sec > 0) state.duration = result.duration_sec;
    ui['result-panel'].setAttribute('aria-busy', 'false');
    ui['result-empty'].hidden = true;
    ui['result-processing'].hidden = true;
    ui['result-content'].hidden = false;
    ui['progress-block'].hidden = true;
    ui['download-button'].hidden = false;
    ui['result-count'].hidden = false;
    ui['result-count'].textContent = isDemo ? 'ДЕМО' : String(state.events.length).padStart(2, '0');
    const positive = result.verdict === 'violation';
    ui['verdict-block'].className = `verdict-block${positive ? '' : ' negative'}`;
    ui['verdict-symbol'].replaceChildren(icon(positive ? 'info' : 'check'));
    ui['verdict-eyebrow'].textContent = isDemo ? 'Пример из исходного датасета · не анализ модели' : 'Проверка завершена';
    const verdict = positive ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено';
    ui['verdict-text'].textContent = verdict;
    ui['result-meta'].replaceChildren();
    const metadata = [];
    if (isDemo) metadata.push('Исходная разметка');
    else if (typeof result.model_name === 'string' && result.model_name) metadata.push(result.model_name);
    if (!isDemo && finiteNumber(result.elapsed_sec)) metadata.push(`Проверка: ${result.elapsed_sec.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} с`);
    if (state.duration > 0) metadata.push(`Видео: ${formatTime(state.duration)}`);
    metadata.forEach((item) => ui['result-meta'].append(textElement('span', '', item)));
    const limitations = Array.isArray(result.limitations) ? result.limitations.filter((item) => typeof item === 'string' && item.trim()) : [];
    ui['result-limitations'].hidden = limitations.length === 0;
    ui['limitations-list'].replaceChildren(...limitations.map((item) => textElement('li', '', item)));
    ui['events-heading'].hidden = !state.events.length;
    ui['events-total'].textContent = eventCount(state.events.length);
    ui['event-list'].replaceChildren(...state.events.map(renderEvent));
    ui['no-events'].hidden = positive;
    ui['timeline-caption'].textContent = state.events.length ? eventCount(state.events.length) : 'События не отмечены';
    ui['timeline-duration'].textContent = formatTime(state.duration || null);
    renderTimeline();
    updateFileMeta();
    updateAnalyzeButton();
    announce(`${isDemo ? 'Пример из исходного датасета. Событие взято из исходной разметки. Модель не запускалась. ' : ''}${verdict}. ${eventCount(state.events.length)}.`);
  }

  function renderTimeline() {
    ui['timeline-markers'].replaceChildren();
    const validTimes = state.events.filter((event) => finiteNumber(event.timecode_sec));
    const length = state.duration || Math.max(1, ...validTimes.map((event) => event.timecode_sec + 1));
    state.events.forEach((event, index) => {
      if (!finiteNumber(event.timecode_sec)) return;
      const marker = textElement('button', 'timeline-marker');
      marker.type = 'button';
      marker.style.left = `${Math.max(0, Math.min(100, (event.timecode_sec / length) * 100))}%`;
      marker.title = `${formatTime(event.timecode_sec)} — ${eventTitle(event.type)}`;
      marker.setAttribute('aria-label', `Событие ${index + 1}, ${formatTime(event.timecode_sec)}: ${eventTitle(event.type)}`);
      marker.addEventListener('click', () => {
        seekTo(event.timecode_sec, index);
        const card = $(`event-${index}`);
        card?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
      });
      ui['timeline-markers'].append(marker);
    });
  }

  function seekTo(seconds, eventIndex) {
    if (!finiteNumber(seconds)) return;
    try { ui['video-player'].currentTime = state.duration ? Math.min(seconds, state.duration) : seconds; }
    catch { announce(`Отметка ${formatTime(seconds)}. Предпросмотр этого формата недоступен; откройте кадры события.`); }
    if (typeof eventIndex === 'number') {
      state.activeEvent = eventIndex;
      ui['event-list'].querySelectorAll('.event-card').forEach((card, index) => card.classList.toggle('active', index === eventIndex));
      ui['timeline-markers'].querySelectorAll('.timeline-marker').forEach((marker) => marker.classList.toggle('active', marker.getAttribute('aria-label').startsWith(`Событие ${eventIndex + 1},`)));
    }
    announce(`Момент видеозаписи: ${formatTime(seconds)}.`);
  }

  async function loadDemo() {
    if (state.configLoading) return;
    if (!state.config) {
      showError('Не удалось подключиться к сервису. Повторите подключение, чтобы открыть пример из исходного датасета.');
      return;
    }
    if (state.config.demo_available === false) {
      const message = 'Пример из исходного датасета пока не подключён. Можно выбрать своё видео для предпросмотра.';
      showError(message);
      announce(message);
      return;
    }
    cancelCurrentWork();
    releasePreview();
    resetResults();
    clearError();
    state.source = 'demo';
    state.file = null;
    state.busy = true;
    state.privacyBusy = true;
    const controller = new AbortController();
    state.privacyAbort = controller;
    ui['video-file'].value = '';
    ui['demo-notice'].hidden = false;
    const generation = state.generation;
    showSelectedFile('Пример из исходного датасета');
    showPrivacyStatus('processing');
    showProcessing('Размываем лица для предпросмотра', 'Готовим пример из исходного датасета. Событие взято из исходной разметки; модель не запускалась.');
    updateAnalyzeButton();
    announce('Размываем лица для предпросмотра демо. Модель анализа событий не запускалась.');
    try {
      const response = await fetch('/api/ui/demo', { credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
      if (response.status === 404 || response.status === 503) throw new Error('demo-unavailable');
      if (!response.ok) throw new Error('demo');
      const data = await response.json();
      if (generation !== state.generation || controller.signal.aborted) return;
      const result = data.result && typeof data.result === 'object' ? data.result : data;
      if (!['violation', 'no_violation'].includes(result.verdict) || !Array.isArray(result.violations)) throw new Error('demo');
      state.duration = finiteNumber(data.duration_sec) ? data.duration_sec : 0;
      displayPreview('/api/ui/demo/video', typeof data.filename === 'string' ? data.filename : 'Пример из исходного датасета.mov');
      renderResult(data, true);
    } catch (error) {
      if (generation !== state.generation || controller.signal.aborted) return;
      showPrivacyStatus('error', 'Пример не показан. Повторите подготовку или вернитесь к загрузке своего видео.');
      showFailure(error.message === 'demo-unavailable'
        ? 'Пример из исходного датасета пока не подключён. Можно выбрать своё видео для предпросмотра.'
        : 'Не удалось открыть пример из исходного датасета. Проверьте соединение и попробуйте ещё раз.');
    } finally {
      if (state.privacyAbort === controller) {
        state.privacyAbort = null;
        state.privacyBusy = false;
        updateAnalyzeButton();
      }
    }
  }

  function downloadResult() {
    if (!state.result) return;
    const result = state.source === 'demo' ? { ...state.result, demo: true, notice: 'Пример из исходного датасета. Событие взято из исходной разметки. Модель не запускалась.' } : state.result;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = state.source === 'demo' ? 'forte-demo-result.json' : `forte-result-${String(state.jobId || 'video').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 80)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  ui['drop-zone'].addEventListener('click', () => { ui['video-file'].value = ''; ui['video-file'].click(); });
  ui['change-button'].addEventListener('click', () => { ui['video-file'].value = ''; ui['video-file'].click(); });
  ui['video-file'].addEventListener('change', (event) => selectFile(event.target.files[0]));
  ui['clear-button'].addEventListener('click', () => { clearVideo(); ui['drop-zone'].focus(); announce('Видеозапись удалена из проверки.'); });
  ui['analyze-button'].addEventListener('click', startAnalysis);
  ui['preview-retry'].addEventListener('click', () => {
    if (state.source === 'demo') loadDemo();
    else preparePreview();
  });
  ui['demo-button'].addEventListener('click', () => {
    if (state.source === 'demo') returnToCheck();
    else loadDemo();
  });
  $('check-video-link').addEventListener('click', (event) => {
    if (state.source !== 'demo') return;
    event.preventDefault();
    returnToCheck();
  });
  ui['config-retry'].addEventListener('click', loadConfig);
  ui['poll-retry'].addEventListener('click', () => {
    if (state.jobId && state.busy) {
      state.pollAbort?.abort();
      state.pollAbort = null;
      pollJob(state.jobId, state.generation);
    }
  });
  ui['download-button'].addEventListener('click', downloadResult);
  $('dismiss-error').addEventListener('click', clearError);
  $('guide-button').addEventListener('click', () => ui['guide-dialog'].showModal());
  ui['seek-evidence-button'].addEventListener('click', () => { ui['evidence-dialog'].close(); seekTo(state.evidenceTime); ui['video-player'].focus(); });
  document.querySelectorAll('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach((dialog) => {
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
  });

  let dragDepth = 0;
  ui['media-workspace'].addEventListener('dragenter', (event) => {
    event.preventDefault();
    dragDepth += 1;
    ui['media-workspace'].classList.add('dragging');
  });
  ui['media-workspace'].addEventListener('dragover', (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'; });
  ui['media-workspace'].addEventListener('dragleave', (event) => {
    event.preventDefault();
    dragDepth -= 1;
    if (dragDepth <= 0) { dragDepth = 0; ui['media-workspace'].classList.remove('dragging'); }
  });
  ui['media-workspace'].addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    ui['media-workspace'].classList.remove('dragging');
    const files = event.dataTransfer?.files;
    if (files?.length > 1) { showError('Добавьте одну видеозапись за раз.'); return; }
    selectFile(files?.[0]);
  });
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());
  ui['video-player'].addEventListener('loadedmetadata', () => {
    if (!state.source || !state.previewReady) return;
    ui['player-shell'].classList.toggle('portrait', ui['video-player'].videoHeight > ui['video-player'].videoWidth);
    if (finiteNumber(ui['video-player'].duration) && ui['video-player'].duration > 0) state.duration = ui['video-player'].duration;
    ui['timeline-duration'].textContent = formatTime(state.duration || null);
    ui['playback-fallback'].hidden = true;
    updateFileMeta();
    if (state.events.length) renderTimeline();
  });
  ui['video-player'].addEventListener('error', () => { if (state.source && state.previewReady) ui['playback-fallback'].hidden = false; });
  ui['video-player'].addEventListener('timeupdate', () => {
    const percent = state.duration ? Math.min(100, Math.max(0, (ui['video-player'].currentTime / state.duration) * 100)) : 0;
    ui['timeline-progress'].style.width = `${percent}%`;
  });
  window.addEventListener('online', () => { if (state.busy && state.jobId && state.connectionFailures) pollJob(state.jobId, state.generation); });
  window.addEventListener('beforeunload', () => state.privacyAbort?.abort());
  loadConfig();
})();
