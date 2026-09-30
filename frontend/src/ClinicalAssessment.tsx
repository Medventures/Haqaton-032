import { useState } from 'react';
import { api } from './types';
import type { ClinicalAssessment as Assessment, Snapshot } from './types';

const fields: [keyof Assessment['values'], string][] = [
  ['diagnosis', 'Предполагаемый диагноз'],
  ['differential', 'Другие возможные причины'],
  ['reasoning', 'Обоснование и ограничения'],
  ['treatment', 'Варианты лечения'],
  ['missing_data', 'Что уточнить и обследовать'],
  ['red_flags', 'Тревожные признаки и срочность'],
];

export function ClinicalAssessment({ snapshot, enabled, busy, update }: {
  snapshot: Snapshot; enabled: boolean; busy: boolean; update: (next: Snapshot) => void;
}) {
  const [requesting, setRequesting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const assessment = snapshot.clinicalAssessment;
  const processing = requesting || snapshot.clinicalStatus === 'processing';
  const stale = !!assessment && (assessment.documentRevision !== snapshot.documentRevision
    || assessment.transcriptRevision !== snapshot.transcriptRevision);
  const hasData = snapshot.transcript.length > 0 || Object.values(snapshot.values).some(value => value?.trim());
  const canAnalyze = enabled && hasData && ['ready', 'stopped'].includes(snapshot.status) && !busy && !processing;

  async function analyze() {
    if (!canAnalyze) return;
    setRequesting(true); setError('');
    try {
      update(await api<Snapshot>(`/sessions/${snapshot.id}/clinical-assessment`, { method: 'POST' }));
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Не удалось выполнить анализ.');
      try { update(await api<Snapshot>(`/sessions/${snapshot.id}`)); } catch { /* Keep the original error. */ }
    } finally { setRequesting(false); }
  }

  async function exportAssessment() {
    setExporting(true); setError('');
    try {
      const result = await api(`/sessions/${snapshot.id}/clinical-assessment/export`);
      const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `clinical-draft-${snapshot.id.slice(0, 8)}.json`; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setError(error instanceof Error ? error.message : 'Не удалось скачать анализ.'); }
    finally { setExporting(false); }
  }

  return <section className="clinical-panel panel" aria-labelledby="clinical-heading" aria-busy={processing}>
    <div className="panel-heading clinical-heading">
      <div className="panel-title"><div><h2 id="clinical-heading">Диагноз и план лечения</h2><p>Клинические гипотезы ассистента · для проверки врачом</p></div></div>
      <div className="clinical-actions">
        {assessment && <button className="button secondary" disabled={exporting} onClick={() => void exportAssessment()}>Скачать анализ</button>}
        <button className="button primary" disabled={!canAnalyze} onClick={() => void analyze()}>
          {processing ? <><span className="spinner" />Анализируем…</> : assessment ? 'Обновить анализ' : 'Предложить диагноз и лечение'}
        </button>
      </div>
    </div>
    <div className="clinical-content">
      <p className="clinical-note">Предварительный вывод ИИ может содержать ошибки. Врач проверяет гипотезы, противопоказания и выбирает назначения перед использованием.</p>
      {!enabled && <p className="clinical-message" role="status">Для анализа подключите языковую модель. Демо-режим заполняет только сведения из разговора.</p>}
      {enabled && !assessment && !processing && <p className="clinical-message">Добавьте жалобы и анамнез. После завершения приёма анализ появится автоматически; по заполненной вручную форме его можно запустить кнопкой.</p>}
      {processing && <p className="clinical-message" role="status">Ассистент анализирует сведения о пациенте и готовит предварительный план…</p>}
      {stale && <p className="clinical-stale" role="status">Данные консультации изменились. Показан предыдущий анализ — обновите его перед использованием.</p>}
      {(error || snapshot.clinicalError) && <div className="error-banner" role="alert">{error || snapshot.clinicalError}</div>}
      <div className="clinical-fields">{fields.map(([key, title]) => <div className={`clinical-field clinical-${key}`} key={key}>
        <label htmlFor={`clinical-${key}`}>{title}</label>
        <textarea id={`clinical-${key}`} readOnly rows={key === 'treatment' ? 5 : 3}
          value={assessment?.values[key] ?? ''}
          placeholder={assessment ? 'Недостаточно сведений или вывод не сформирован' : 'Появится после анализа консультации'} />
      </div>)}</div>
      {assessment && <p className="clinical-source">Источник: {assessment.provider} · {assessment.model} · {new Date(assessment.generatedAt).toLocaleString('ru-RU')} · Требует проверки врачом</p>}
    </div>
  </section>;
}
