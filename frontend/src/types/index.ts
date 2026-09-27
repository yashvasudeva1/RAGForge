/**
 * TypeScript types for RAGForge frontend, aligned with backend Pydantic models.
 */

export type PortType =
  | 'document'
  | 'documents'
  | 'chunk'
  | 'chunks'
  | 'embedding'
  | 'embeddings'
  | 'embedded_chunk'
  | 'embedded_chunks'
  | 'query'
  | 'context'
  | 'prompt'
  | 'answer'
  | 'retrieval_results'
  | 'generation_result'
  | 'evaluation_result'
  | 'any';

export interface NodePortDefinition {
  name: string;
  port_type: PortType;
  required: boolean;
  description?: string;
}

export type FieldType =
  | 'string'
  | 'integer'
  | 'float'
  | 'boolean'
  | 'select'
  | 'multi_select'
  | 'textarea'
  | 'file_path'
  | 'secret'
  | 'json';

export interface NodeField {
  name: string;
  field_type: FieldType;
  label: string;
  description: string;
  default: any;
  required: boolean;
  options?: string[];
  placeholder?: string;
  min_val?: number;
  max_val?: number;
  step?: number;
}

export interface NodeMetadata {
  name: string;
  description: string;
  category: string;
  icon: string;
  tags: string[];
}

export interface NodeSchema {
  type: string;
  metadata: NodeMetadata;
  category: string;
  input_ports: NodePortDefinition[];
  output_ports: NodePortDefinition[];
  fields: NodeField[];
}

export interface NodePosition {
  x: number;
  y: number;
}

export interface PipelineNodeConfig {
  id: string;
  type: string;
  label?: string;
  position: NodePosition;
  config: Record<string, any>;
}

export interface PipelineEdgeConfig {
  id: string;
  source_node_id: string;
  source_port: string;
  target_node_id: string;
  target_port: string;
}

export interface PipelineConfig {
  id: string;
  name: string;
  description: string;
  nodes: PipelineNodeConfig[];
  edges: PipelineEdgeConfig[];
  tags: string[];
  is_template?: boolean;
}

export type NodeExecutionStatus = 'idle' | 'waiting' | 'running' | 'completed' | 'failed' | 'skipped';

export interface NodeExecutionRecord {
  node_id: string;
  node_type: string;
  status: NodeExecutionStatus;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  error?: string;
  output_preview?: Record<string, any>;
}

export interface PipelineExecutionRecord {
  execution_id: string;
  pipeline_id: string;
  pipeline_name: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  error?: string;
  node_records: Record<string, NodeExecutionRecord>;
}

export interface ExecutionEvent {
  type:
    | 'pipeline_started'
    | 'node_started'
    | 'node_completed'
    | 'node_failed'
    | 'node_skipped'
    | 'pipeline_completed'
    | 'pipeline_failed';
  execution_id: string;
  node_id?: string;
  node_type?: string;
  timestamp: string;
  data: Record<string, any>;
}

export interface CategoryInfo {
  id: string;
  label: string;
  color: string;
  node_count: number;
}
