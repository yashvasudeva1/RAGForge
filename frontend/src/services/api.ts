import {
  CategoryInfo,
  NodeSchema,
  PipelineConfig,
  PipelineExecutionRecord,
} from '../types';

const API_BASE = '/api';

export async function fetchNodeSchemas(): Promise<NodeSchema[]> {
  const res = await fetch(`${API_BASE}/nodes`);
  if (!res.ok) throw new Error('Failed to load node schemas');
  return res.json();
}

export async function fetchCategories(): Promise<CategoryInfo[]> {
  const res = await fetch(`${API_BASE}/nodes/categories`);
  if (!res.ok) throw new Error('Failed to load categories');
  return res.json();
}

export async function fetchTemplates(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/pipelines/templates`);
  if (!res.ok) throw new Error('Failed to load templates');
  return res.json();
}

export async function fetchPipelines(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/pipelines`);
  if (!res.ok) throw new Error('Failed to load pipelines');
  return res.json();
}

export async function fetchPipeline(id: string): Promise<PipelineConfig> {
  const res = await fetch(`${API_BASE}/pipelines/${id}`);
  if (!res.ok) throw new Error(`Failed to load pipeline: ${id}`);
  return res.json();
}

export async function savePipeline(pipeline: PipelineConfig): Promise<PipelineConfig> {
  const res = await fetch(`${API_BASE}/pipelines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pipeline),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || 'Failed to save pipeline');
  }
  return res.json();
}

export async function validatePipeline(pipeline: PipelineConfig): Promise<{
  is_valid: boolean;
  errors: string[];
  warnings: Array<{ code: string; message: string; node_id?: string }>;
}> {
  const res = await fetch(`${API_BASE}/pipelines/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pipeline),
  });
  if (!res.ok) throw new Error('Validation request failed');
  return res.json();
}

export async function executePipelineDirect(
  pipeline: PipelineConfig,
  inputs: Record<string, any> = {}
): Promise<PipelineExecutionRecord> {
  const res = await fetch(`${API_BASE}/executions/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pipeline, inputs }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || 'Execution failed');
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<{
  filename: string;
  file_path: string;
  size_bytes: number;
}> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}
