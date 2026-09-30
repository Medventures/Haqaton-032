import { useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { SessionConnection } from './audio';
import { describeProviders } from './provider-config.js';
import { ClinicalAssessment } from './ClinicalAssessment';
import { api, ApiError, validateSchema } from './types';
import type { Config, FieldMeta, FieldSchema, FieldValue, FormSchema, Snapshot } from './types';

type IconName = 'plus' | 'file' | 'mic' | 'stop' | 'settings' | 'download' | 'spark' | 'check' | 'chevron' | 'close' | 'wave' | 'lock' | 'refresh' | 'info';
const paths: Record<IconName, string> = {
  plus: 'M12 5v14M5 12h14', file: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8M8 17h5',
  mic: 'M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3ZM5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8',
  stop: 'M7 7h10v10H7z', settings: 'M4 7h16M4 17h16M9 4v6M15 14v6', download: 'M12 3v12M7 10l5 5 5-5M5 16v4a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-4',
  spark: 'm12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z', check: 'm5 12 4 4L19 6', chevron: 'm9 5 7 7-7 7',
  close: 'm6 6 12 12M18 6 6 18', wave: 'M3 10v4M7 6v12M12 3v18M17 6v12M21 10v4', lock: 'M6 10h12v11H6zM8 10V6a4 4 0 0 1 8 0v4',
  refresh: 'M20 7v5h-5M4 17v-5h5M6.1 6.1A8 8 0 0 1 20 12M4 12a8 8 0 0 0 13.9 5.9', info: 'M12 8h.01M12 11v6M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
};
function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Что-то пошло не так. Попробуйте ещё раз.';
const clock = (ms: number) => `${Math.floor(ms / 60000).toString().padStart(2, '0')}:${Math.floor(ms / 1000 % 60).toString().padStart(2, '0')}`;
const emptyMeta: FieldMeta = { revision: 0, source: 'empty', locked: false };
const speakerLabel = (speaker: string | null) => speaker === 'doctor' ? 'Врач' : speaker === 'patient' ? 'Пациент' : speaker ? /^S?\d+$/i.test(speaker) ? `Участник ${speaker.replace(/^S/i, '')}` : speaker : 'Участник';

function FormField({ id, schema, value, meta, sessionId, update, report }: {
  id: string; schema: FieldSchema; value: FieldValue; meta: FieldMeta; sessionId: string;
  update: (snapshot: Snapshot) => void; report: (message: string) => void;
}) {
  const [draft, setDraft] = useState(value ?? '');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [baseRevision, setBaseRevision] = useState(meta.revision);
  const [conflict, setConflict] = useState(false);
  useEffect(() => { if (!dirty) { setDraft(value ?? ''); setBaseRevision(meta.revision); } }, [value, meta.revision, dirty]);

  async function save(next: FieldValue) {
    if (saving) return;
    setSaving(true);
    try {
      const snapshot = await api<Snapshot>(`/sessions/${sessionId}/fields/${encodeURIComponent(id)}`, {
        method: 'PATCH', body: JSON.stringify({ value: next, expectedRevision: baseRevision }),
      });
      setDirty(false); setConflict(false); update(snapshot);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setConflict(true);
        try {
          const snapshot = await api<Snapshot>(`/sessions/${sessionId}`);
          update(snapshot);
          setBaseRevision(snapshot.fieldMeta[id]?.revision ?? 0);
        } catch (refreshError) { report(errorMessage(refreshError)); }
      } else report(errorMessage(error));
    } finally { setSaving(false); }
  }

  async function unlock() {
    setSaving(true);
    try {
      update(await api<Snapshot>(`/sessions/${sessionId}/fields/${encodeURIComponent(id)}/unlock`, {
        method: 'POST', body: JSON.stringify({ expectedRevision: meta.revision }),
      }));
    } catch (error) { report(errorMessage(error)); }
    finally { setSaving(false); }
  }

  const label = schema.title || id;
  const suggestionPresent = Object.prototype.hasOwnProperty.call(meta, 'suggestion') && meta.suggestion !== undefined && meta.suggestion !== value;
  return <div className={`form-field ${meta.locked ? 'field-manual' : ''}`}>
    <div className="field-title-row">
      <label id={`label-${id}`} htmlFor={`field-${id}`} title={schema.description}>{label}</label>
      <span className={`field-origin ${meta.source === 'llm' ? 'origin-ai' : ''}`}>
        {saving ? 'Сохраняем…' : dirty ? 'Не сохранено' : meta.locked ? <><Icon name="lock" size={12} />Ваша правка</> : meta.source === 'llm' ? <><Icon name="spark" size={13} />Из разговора</> : ''}
      </span>
    </div>
    {schema.enum ? <div className="radio-group" role="radiogroup" aria-labelledby={`label-${id}`}>
      {schema.enum.filter((option): option is string => option !== null).map(option => <label className={`radio-option ${value === option ? 'selected' : ''}`} key={option}>
        <input type="radio" name={id} value={option} checked={value === option} disabled={saving} onChange={() => { setDraft(option); setDirty(true); void save(option); }} />
        <span>{schema['x-enumLabels']?.[option] || option}</span>
      </label>)}
      {value !== null && <button type="button" className="text-button clear-option" disabled={saving} onClick={() => { setDraft(''); setDirty(true); void save(null); }}>Очистить</button>}
    </div> : schema['x-ui'] === 'textarea' ? <textarea id={`field-${id}`} value={draft} disabled={saving} maxLength={schema.maxLength} placeholder="Заполнится по разговору или введите текст" rows={3} onChange={event => { setDraft(event.target.value); setDirty(true); }} onBlur={() => { if (dirty && !conflict) void save(draft.trim() ? draft : null); }} />
      : <input id={`field-${id}`} type="text" value={draft} disabled={saving} maxLength={schema.maxLength} placeholder="Пока не указано" onChange={event => { setDraft(event.target.value); setDirty(true); }} onBlur={() => { if (dirty && !conflict) void save(draft.trim() ? draft : null); }} />}
    <div className="field-footnote"><span>{schema.enum && value === null ? 'Пока не обсуждалось' : ''}</span>{meta.locked && <button type="button" className="text-button" disabled={saving || dirty} onClick={() => void unlock()}>Включить автозаполнение</button>}</div>
    {conflict && <div className="field-conflict"><span>Поле обновилось во время редактирования. Ваша правка сохранена в редакторе.</span><div><button className="text-button" disabled={saving} onClick={() => void save(draft.trim() ? draft : null)}>Сохранить мою правку</button><button className="text-button" onClick={() => { setDirty(false); setConflict(false); }}>Принять актуальную</button></div></div>}
    {suggestionPresent && !dirty && <div className="suggestion"><Icon name="spark" size={15} /><div><strong>Новое из разговора</strong><p>{meta.suggestion === null ? 'Очистить поле' : schema['x-enumLabels']?.[meta.suggestion!] || meta.suggestion}</p></div><button className="text-button" disabled={saving} onClick={() => void save(meta.suggestion ?? null)}>Принять</button></div>}
  </div>;
}

function SchemaEditor({ schema, close, apply }: { schema: FormSchema; close: () => void; apply: (schema: FormSchema) => Promise<void> }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [raw, setRaw] = useState(JSON.stringify(schema, null, 2));
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  useEffect(() => { dialog.current?.showModal(); }, []);
  async function submit() {
    setError(''); setSaving(true);
    try { await apply(validateSchema(raw)); close(); }
    catch (error) { setError(errorMessage(error)); }
    finally { setSaving(false); }
  }
  return <dialog className="schema-dialog" ref={dialog} onCancel={close}>
    <div className="dialog-heading"><div><span className="eyebrow">КОНСТРУКТОР</span><h2>Описание формы</h2></div><button className="icon-button" aria-label="Закрыть настройки" onClick={close}><Icon name="close" /></button></div>
    <p className="dialog-copy">Одна JSON Schema задаёт поля для ассистента и интерфейса. Строка становится полем ввода, <code>enum</code> — выбором одного варианта. Значение <code>null</code> означает, что поле ещё не заполнено.</p>
    <label className="sr-only" htmlFor="schema-source">JSON Schema формы</label><textarea id="schema-source" className="schema-source" spellCheck={false} value={raw} onChange={event => setRaw(event.target.value)} />
    <div className="schema-hints"><code>title</code> — подпись · <code>description</code> — инструкция LLM · <code>x-ui: textarea</code> — многострочный ввод · <code>x-enumLabels</code> — подписи вариантов</div>
    {error && <div className="error-banner" role="alert">{error}</div>}
    <div className="dialog-footer"><p>Применение создаст новую консультацию.</p><button className="button secondary" onClick={close} disabled={saving}>Отмена</button><button className="button primary" onClick={() => void submit()} disabled={saving}>{saving ? 'Создаём…' : 'Применить форму'}</button></div>
  </dialog>;
}

function TranscriptInput({ close, apply, llmLabel, llmDemo }: {
  close: () => void; apply: (text: string) => Promise<void>; llmLabel: string; llmDemo: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  useEffect(() => { dialog.current?.showModal(); }, []);
  async function submit() {
    if (!text.trim() || sending) return;
    setSending(true); setError('');
    try { await apply(text.trim()); close(); }
    catch (error) { setError(errorMessage(error)); }
    finally { setSending(false); }
  }
  return <dialog className="schema-dialog transcript-dialog" ref={dialog} aria-labelledby="transcript-dialog-title" onCancel={event => { if (sending) event.preventDefault(); else close(); }}>
    <div className="dialog-heading"><div><span className="eyebrow">ГОТОВЫЙ РАЗГОВОР</span><h2 id="transcript-dialog-title">Заполнить форму по тексту</h2></div><button className="icon-button" aria-label="Закрыть ввод текста" disabled={sending} onClick={close}><Icon name="close" /></button></div>
    <p className="dialog-copy">Вставьте текст консультации или уточнение к уже добавленному разговору. Подпишите реплики «Врач:» и «Пациент:», если известны роли участников.</p>
    <label className="input-label" htmlFor="transcript-source">Текст разговора</label>
    <textarea id="transcript-source" className="transcript-source" autoFocus value={text} maxLength={20000} disabled={sending} onChange={event => setText(event.target.value)} placeholder="Вставьте расшифровку разговора врача и пациента…" />
    <div className="transcript-input-meta"><span>{llmDemo ? 'Демонстрационное заполнение без модели' : `Форму заполнит ${llmLabel}`}</span><span>{text.length.toLocaleString('ru-RU')} / 20 000</span></div>
    {error && <div className="error-banner" role="alert">{error}</div>}
    <div className="dialog-footer"><p>{sending ? 'Текст принят в обработку. Ожидаем заполнение формы…' : 'Текст добавится в текущую консультацию.'}</p><button className="button secondary" disabled={sending} onClick={close}>Отмена</button><button className="button primary" disabled={sending || !text.trim()} onClick={() => void submit()}>{sending ? <><span className="spinner" />Заполняем…</> : <><Icon name="spark" size={16} />Заполнить форму</>}</button></div>
  </dialog>;
}

export default function App() {
  const [config, setConfig] = useState<Config>();
  const [defaultSchema, setDefaultSchema] = useState<FormSchema>();
  const [snapshot, setSnapshot] = useState<Snapshot>();
  const [initializing, setInitializing] = useState(true);
  const [busy, setBusy] = useState(false);
  const [recordingAction, setRecordingAction] = useState<'start' | 'stop' | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [partial, setPartial] = useState('');
  const [connection, setConnection] = useState('connecting');
  const [microphone, setMicrophone] = useState(false);
  const [level, setLevel] = useState(0);
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [transcriptInputOpen, setTranscriptInputOpen] = useState(false);
  const [startTime, setStartTime] = useState<number>();
  const [elapsed, setElapsed] = useState(0);
  const socket = useRef<SessionConnection | undefined>(undefined);
  const activeId = useRef<string | undefined>(undefined);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const bootStarted = useRef(false);
  const providers = describeProviders(config);

  const update = useCallback((next: Snapshot) => {
    if (activeId.current && activeId.current !== next.id) return;
    setSnapshot(previous => {
      if (previous?.id === next.id && (next.documentRevision < previous.documentRevision || next.transcriptRevision < previous.transcriptRevision || (next.clinicalRevision ?? 0) < (previous.clinicalRevision ?? 0))) return previous;
      return next;
    });
    if (next.status === 'stopped') setPartial('');
  }, []);

  const createSession = useCallback(async (schema: FormSchema) => {
    const next = await api<Snapshot>('/sessions', { method: 'POST', body: JSON.stringify({ formSchema: schema }) });
    activeId.current = next.id;
    localStorage.setItem('consultation-session-id', next.id);
    setSnapshot(next); setPartial(''); setError(''); setNotice(''); setElapsed(0); setStartTime(undefined); setMicrophone(false); setLevel(0);
  }, []);

  const initialize = useCallback(async () => {
    setInitializing(true); setError('');
    try {
      const [settings, schema] = await Promise.all([api<Config>('/config'), api<FormSchema>('/forms/default')]);
      setConfig(settings); setDefaultSchema(schema);
      const saved = localStorage.getItem('consultation-session-id');
      if (saved) {
        try {
          const previous = await api<Snapshot>(`/sessions/${encodeURIComponent(saved)}`);
          activeId.current = previous.id; setSnapshot(previous);
          if (previous.status === 'recording') setNotice('Восстановлена открытая консультация. Микрофон после обновления страницы выключен; завершите эту запись перед началом новой.');
        } catch (error) {
          if (error instanceof ApiError && error.status === 404) await createSession(schema);
          else throw error;
        }
      } else await createSession(schema);
    } catch (error) { setError(errorMessage(error)); }
    finally { setInitializing(false); }
  }, [createSession]);

  useEffect(() => {
    if (bootStarted.current) return;
    bootStarted.current = true;
    void initialize();
  }, [initialize]);
  useEffect(() => {
    if (!snapshot?.id) return;
    const transport = new SessionConnection(snapshot.id, { snapshot: update, partial: setPartial, connection: setConnection, error: setError, level: setLevel, microphone: setMicrophone });
    socket.current = transport;
    return () => transport.close();
  }, [snapshot?.id, update]);
  useEffect(() => { transcriptEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [snapshot?.transcriptRevision, partial]);
  useEffect(() => {
    if (!startTime || !microphone || snapshot?.status !== 'recording') return;
    const timer = window.setInterval(() => setElapsed(Date.now() - startTime), 1000);
    return () => window.clearInterval(timer);
  }, [startTime, microphone, snapshot?.status]);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (microphone) { event.preventDefault(); event.returnValue = ''; }
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [microphone]);

  async function act(action: () => Promise<void>) {
    setBusy(true); setError(''); setNotice('');
    try { await action(); } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  async function start() {
    if (!snapshot || !config || providers.asrDemo || busy) return;
    await act(async () => {
      setRecordingAction('start');
      try {
        if (!socket.current) throw new Error('Дождитесь подключения к серверу и повторите начало приёма.');
        await socket.current.startMicrophone();
        setStartTime(Date.now());
      } finally { setRecordingAction(null); }
    });
  }
  async function stop() {
    if (!snapshot) return;
    await act(async () => {
      setRecordingAction('stop');
      try {
        await socket.current?.finishAudio();
        update(await api<Snapshot>(`/sessions/${snapshot.id}/stop`, { method: 'POST' }));
      } finally { setRecordingAction(null); }
    });
  }
  async function submitTranscript(text: string) {
    if (!snapshot) throw new Error('Сначала создайте консультацию.');
    setBusy(true); setError(''); setNotice('');
    try {
      update(await api<Snapshot>(`/sessions/${snapshot.id}/transcript`, {
        method: 'POST', body: JSON.stringify({ text, speaker: null }),
      }));
    } finally { setBusy(false); }
  }
  async function exportJson() {
    if (!snapshot) return;
    await act(async () => {
      const values = await api<Record<string, FieldValue>>(`/sessions/${snapshot.id}/export`);
      const url = URL.createObjectURL(new Blob([JSON.stringify(values, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `consultation-${snapshot.id.slice(0, 8)}.json`; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setNotice('Значения формы сохранены в JSON.');
    });
  }

  const recording = snapshot?.status === 'recording';
  const processing = snapshot?.status === 'processing';
  const finished = snapshot?.status === 'stopped';
  const entries = Object.entries(snapshot?.formSchema.properties || {});
  const filled = Object.values(snapshot?.values || {}).filter(value => value !== null && value !== '').length;
  const transcript = snapshot?.transcript || [];
  const duration = Math.max(elapsed, ...transcript.map(segment => segment.endMs || 0), 0);
  const date = new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' });
  const canPaste = !!snapshot && ['ready', 'stopped'].includes(snapshot.status) && !busy;
  const currentSchema = snapshot?.formSchema;
  // Existing sessions retain their fixed schema; new consultations use the updated
  // default when the previous template is exactly the older built-in form.
  const legacyDefault = !!currentSchema && !!defaultSchema
    && currentSchema.title === defaultSchema.title
    && currentSchema.description === defaultSchema.description
    && !currentSchema.properties.diagnosis
    && Object.keys(currentSchema.properties).length === Object.keys(defaultSchema.properties).length - 1
    && Object.entries(currentSchema.properties).every(([key, field]) => JSON.stringify(field) === JSON.stringify(defaultSchema.properties[key]));
  const newSessionSchema = legacyDefault ? defaultSchema! : currentSchema || defaultSchema!;
  const recordingTitle = initializing ? 'Подготавливаем рабочее место…' : snapshot?.status === 'error' ? 'Обработка разговора прервана' : recordingAction === 'start' ? 'Подключаем микрофон…' : recordingAction === 'stop' ? 'Завершаем приём…' : recording ? microphone ? 'Идёт приём — слушаем разговор' : 'Микрофон выключен' : processing ? 'Заполняем форму по разговору' : finished ? 'Черновик готов к проверке' : providers.asrDemo ? 'Подключите распознавание речи' : 'Готовы к голосовому приёму';
  const recordingDescription = snapshot?.status === 'error' ? 'Сохранённые сведения доступны в форме. Для новой записи создайте консультацию.' : recordingAction === 'stop' || processing ? 'Обрабатываем последние фрагменты разговора и заполняем поля. Дождитесь завершения.' : finished ? 'Проверьте формулировки и при необходимости отредактируйте поля.' : providers.asrDemo ? 'Для голосового приёма настройте сервис распознавания на сервере. Пока можно ввести свой текст.' : recording && !microphone && recordingAction !== 'start' ? 'Завершите этот приём, чтобы обработать сохранённый разговор и начать новый.' : config?.asrProvider === 'openai-compatible' ? 'Говорите в микрофон. Расшифровка и поля будут обновляться фрагментами с небольшой задержкой.' : 'Запишите разговор врача и пациента — важные сведения появятся в форме.';

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="Лист, главная"><span className="brand-mark"><Icon name="plus" size={24} /></span><span>лист<span className="brand-dot">.</span></span></a><span className="brand-divider" /><span className="product-name">Ассистент консультации</span><div className="topbar-right"><span className={`connection ${connection === 'connected' ? 'online' : ''}`}><i />{connection === 'connected' ? 'На связи' : connection === 'reconnecting' ? 'Восстанавливаем связь' : 'Подключаемся'}</span><span className="avatar" aria-label="Рабочее место врача">В</span></div></header>
    <main>
      <div className="breadcrumb"><span>Рабочее пространство</span><Icon name="chevron" size={13} /><span>Консультация</span></div>
      <div className="page-heading"><div><div className="heading-kicker">МЕНЬШЕ ЗАПИСЕЙ. БОЛЬШЕ ВНИМАНИЯ.</div><h1>Лист консультации</h1><p>Разговор становится черновиком. Последнее слово — за врачом.</p></div><div className="page-actions"><button className="button secondary" disabled={!snapshot || busy || recording || processing} onClick={() => void act(() => createSession(newSessionSchema))}><Icon name="plus" size={17} />Новая</button><button className="button secondary" aria-label="Скачать JSON" disabled={!snapshot || busy} onClick={() => void exportJson()}><Icon name="download" size={17} /><span>Скачать JSON</span></button></div></div>

      {error && <div className="error-banner" role="alert"><Icon name="info" size={19} /><span>{error}</span>{!snapshot && !initializing && <button className="text-button" onClick={() => void initialize()}>Повторить</button>}<button className="icon-button" aria-label="Скрыть ошибку" onClick={() => setError('')}><Icon name="close" size={16} /></button></div>}
      {snapshot?.error && snapshot.error !== error && <div className="error-banner" role="alert"><Icon name="info" size={19} /><span>{snapshot.error}</span></div>}
      {notice && <div className="notice-banner" role="status"><Icon name="info" size={17} /><span>{notice}</span><button className="icon-button" aria-label="Скрыть уведомление" onClick={() => setNotice('')}><Icon name="close" size={15} /></button></div>}

      <section className={`recording-card ${recording ? 'is-recording' : ''}`} aria-label="Управление записью">
        <div className="recording-symbol"><Icon name={finished ? 'check' : recording ? 'wave' : 'mic'} size={27} /></div>
        <div className="recording-copy"><div className="recording-title"><h2>{recordingTitle}</h2><span className={`mode-badge ${providers.asrDemo ? 'unavailable' : ''}`}>{providers.asrDemo ? 'НЕ ПОДКЛЮЧЁН' : 'МИКРОФОН'}</span></div><p>{recordingDescription}</p>{config && <div className="provider-summary"><span><Icon name="spark" size={12} />Заполнение: <strong>{providers.llmLabel}</strong></span><span>{providers.asrLabel}</span></div>}</div>
        <div className="recording-controls">{(recording || processing || finished) && <div className="recording-meter"><div className={`wave-bars ${recording ? 'animated' : ''}`} aria-hidden="true">{[8, 15, 24, 14, 29, 18, 11].map((height, index) => <i key={index} style={{ '--bar-height': `${height}px`, '--bar-delay': `${index * -0.13}s`, '--level': microphone ? Math.max(0.25, level) : 1 } as CSSProperties} />)}</div><span className="timer">{clock(duration)}</span></div>}
          {recording ? <button className="button stop-button" disabled={busy} onClick={() => void stop()}>{recordingAction === 'stop' ? <span className="spinner" /> : <Icon name="stop" size={17} />}{recordingAction === 'stop' ? 'Завершаем…' : 'Завершить приём'}</button> : processing ? <span className="processing-label"><span className="spinner" />Обрабатываем</span> : finished ? <span className="finished-label"><Icon name="check" size={17} />Приём завершён</span> : <button className="button primary" disabled={!snapshot || !config || providers.asrDemo || busy || connection !== 'connected' || snapshot.status === 'error'} onClick={() => void start()}>{recordingAction === 'start' ? <span className="spinner" /> : <Icon name="mic" size={18} />}{recordingAction === 'start' ? 'Подключаем…' : 'Начать приём'}</button>}
        </div>
      </section>

      <div className="workspace-grid">
        <section className="document-panel panel"><div className="panel-heading"><div className="panel-title"><span className="panel-icon"><Icon name="file" size={20} /></span><div><h2>{snapshot?.formSchema.title || 'Лист консультации'}</h2><p>{date}<span className="small-separator">·</span>Черновик</p></div></div><button className="button quiet schema-button" aria-label="Настроить форму" disabled={!snapshot || busy || recording || processing} onClick={() => setSchemaOpen(true)}><Icon name="settings" size={17} /><span>Настроить форму</span></button></div>
          <div className="document-status"><span><span className="tiny-dot" />{processing ? 'Ассистент сверяет последние реплики' : recording ? 'Заполняется по ходу разговора' : finished ? 'Проверьте результат перед использованием' : 'Можно заполнять и редактировать вручную'}</span><span>{filled}<span className="count-muted"> / {entries.length} полей</span></span></div>
          <div className="form-content">{initializing && !snapshot ? <div className="skeleton-fields">{[1, 2, 3].map(index => <div key={index}><i /><span /></div>)}</div> : entries.map(([id, field]) => <FormField key={`${snapshot!.id}-${id}`} id={id} schema={field} value={snapshot!.values[id] ?? null} meta={snapshot!.fieldMeta[id] || emptyMeta} sessionId={snapshot!.id} update={update} report={setError} />)}</div>
          <div className="document-footer"><Icon name="check" size={15} /><span>Ручные правки сохраняются отдельно и защищены от перезаписи ассистентом.</span></div>
        </section>

        <aside className="transcript-panel panel"><div className="panel-heading"><div className="panel-title"><span className="panel-icon"><Icon name="wave" size={19} /></span><div><h2>Разговор</h2><p>Расшифровка по мере записи</p></div></div><span className="transcript-counter">{transcript.length}</span></div><div className="transcript-body" aria-label="Расшифровка разговора" aria-live="polite" aria-relevant="additions text">
          {!transcript.length && !partial ? <div className="transcript-empty"><div className="empty-orbit"><span /><div><Icon name="wave" size={30} /></div><span /></div><h3>{microphone ? 'Слушаем разговор' : processing ? 'Распознаём последние фрагменты' : 'Здесь появится разговор'}</h3><p>{providers.asrDemo ? 'Распознавание речи ещё не подключено. Можно заполнить форму по своему тексту.' : microphone ? 'Говорите в микрофон. Первые реплики появятся после обработки фрагмента записи.' : processing ? 'Дождитесь расшифровки и заполнения формы.' : 'Начните приём и разрешите доступ к микрофону. Реплики будут появляться здесь по ходу разговора.'}</p></div> : <div className="transcript-messages">{transcript.map((segment, index) => <article className="transcript-message" key={segment.id}><div className="message-meta"><span className={`speaker-dot ${index % 2 ? 'alternate' : ''}`} /><strong>{speakerLabel(segment.speaker)}</strong><time>{clock(segment.startMs)}</time></div><p>{segment.text}</p></article>)}{partial && <article className="transcript-message partial"><div className="message-meta"><span className="tiny-dot" /><strong>Распознаём…</strong></div><p>{partial}<span className="typing-caret" /></p></article>}<div ref={transcriptEnd} /></div>}
        </div>{providers.asrDemo && <div className="transcript-input-action"><button className="button secondary" disabled={!canPaste} onClick={() => setTranscriptInputOpen(true)}><Icon name="plus" size={16} />Вставить текст разговора</button></div>}<div className="transcript-footer"><Icon name="info" size={15} /><span>{providers.asrDemo ? providers.llmDemo ? 'Текстовый источник · демо без модели' : `Текстовый источник · заполнение ${providers.llmLabel}` : 'Расшифровка может содержать ошибки. Проверяйте важные сведения.'}</span></div></aside>
      </div>
      {snapshot && <ClinicalAssessment key={snapshot.id} snapshot={snapshot} enabled={!!config && !providers.llmDemo} busy={busy} update={update} />}
      <footer className="page-footer"><span>Лист<span className="footer-dot">.</span> Помощник на приёме</span><span>Рабочее место консультации</span></footer>
    </main>
    {schemaOpen && snapshot && <SchemaEditor schema={snapshot.formSchema} close={() => setSchemaOpen(false)} apply={createSession} />}
    {transcriptInputOpen && snapshot && <TranscriptInput close={() => setTranscriptInputOpen(false)} apply={submitTranscript} llmLabel={providers.llmLabel} llmDemo={providers.llmDemo} />}
  </div>;
}



