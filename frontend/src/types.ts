export type FieldValue = string | null;
export interface FieldSchema {
  type: 'string' | ['string', 'null'] | ['null', 'string'];
  title?: string;
  description?: string;
  enum?: FieldValue[];
  maxLength?: number;
  'x-enumLabels'?: Record<string, string>;
  'x-ui'?: 'textarea' | 'input';
}
export interface FormSchema {
  type: 'object';
  title?: string;
  description?: string;
  additionalProperties: false;
  properties: Record<string, FieldSchema>;
  required?: string[];
}
export interface FieldMeta {
  revision: number;
  source: 'empty' | 'llm' | 'doctor';
  locked: boolean;
  suggestion?: FieldValue;
}
export interface Segment {
  id: string;
  revision: number;
  text: string;
  startMs: number;
  endMs: number;
  speaker: string | null;
}
export interface Snapshot {
  id: string;
  formSchema: FormSchema;
  values: Record<string, FieldValue>;
  fieldMeta: Record<string, FieldMeta>;
  transcript: Segment[];
  status: 'ready' | 'recording' | 'processing' | 'stopped' | 'error';
  documentRevision: number;
  transcriptRevision: number;
  error: string | null;
  clinicalAssessment?: ClinicalAssessment | null;
  clinicalStatus?: 'idle' | 'processing' | 'ready' | 'error';
  clinicalError?: string | null;
  clinicalRevision?: number;
}
export interface ClinicalAssessment {
  values: Record<'diagnosis' | 'differential' | 'reasoning' | 'treatment' | 'missing_data' | 'red_flags', FieldValue>;
  provider: string;
  model: string;
  transcriptRevision: number;
  documentRevision: number;
  generatedAt: string;
}
export interface Config {
  asrProvider: string;
  llmProvider: string;
  /** Describes the speech source only; the LLM can independently be real. */
  demoMode: boolean;
  llmDemoMode?: boolean;
  asrLanguage?: string;
}

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${url}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message || body?.message || `Ошибка запроса (${response.status})`;
    throw new ApiError(message, response.status);
  }
  return body as T;
}

export function validateSchema(raw: string): FormSchema {
  const schema = JSON.parse(raw) as FormSchema;
  if (schema?.type !== 'object' || schema.additionalProperties !== false || !schema.properties || Array.isArray(schema.properties)) {
    throw new Error('Нужна JSON Schema: type: "object", additionalProperties: false и объект properties.');
  }
  const fields = Object.entries(schema.properties);
  if (!fields.length) throw new Error('Добавьте хотя бы одно поле в properties.');
  for (const [id, field] of fields) {
    if (!field || typeof field !== 'object') throw new Error(`Поле «${id}»: описание должно быть объектом.`);
    const types = Array.isArray(field.type) ? field.type : [field.type];
    if (!types.includes('string') || types.some(type => !['string', 'null'].includes(type))) {
      throw new Error(`Поле «${id}»: поддерживаются строки, включая nullable string.`);
    }
    if (field.enum && (!Array.isArray(field.enum) || field.enum.some(value => value !== null && typeof value !== 'string') || !field.enum.some(value => typeof value === 'string'))) {
      throw new Error(`Поле «${id}»: enum должен содержать строковые варианты и может содержать null.`);
    }
  }
  return schema;
}
