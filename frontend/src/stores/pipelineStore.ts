import { create } from 'zustand';
import {
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from '@xyflow/react';
import {
  CategoryInfo,
  ExecutionEvent,
  NodeExecutionStatus,
  NodeSchema,
  PipelineConfig,
  PipelineExecutionRecord,
} from '../types';
import {
  executePipelineDirect,
  fetchCategories,
  fetchNodeSchemas,
  savePipeline,
  validatePipeline,
} from '../services/api';
import { ExecutionWebSocketClient } from '../services/websocket';

export interface RAGFlowNodeData extends Record<string, unknown> {
  schema: NodeSchema;
  config: Record<string, any>;
  label?: string;
  executionStatus?: NodeExecutionStatus;
}

export type RAGFlowNode = Node<RAGFlowNodeData>;

interface PipelineState {
  // Discovery
  schemas: NodeSchema[];
  categories: CategoryInfo[];
  isLoadingSchemas: boolean;
  loadSchemas: () => Promise<void>;

  // Pipeline Data
  pipelineId: string;
  pipelineName: string;
  pipelineDescription: string;
  tags: string[];
  nodes: RAGFlowNode[];
  edges: Edge[];
  selectedNodeId: string | null;

  // React Flow Handlers
  onNodesChange: (changes: NodeChange<RAGFlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  setSelectedNodeId: (id: string | null) => void;

  // Node Manipulation
  addNode: (type: string, position: { x: number; y: number }) => void;
  updateNodeConfig: (nodeId: string, key: string, value: any) => void;
  updateNodeLabel: (nodeId: string, label: string) => void;
  removeNode: (nodeId: string) => void;

  // Pipeline Actions
  setPipelineName: (name: string) => void;
  setPipelineDescription: (desc: string) => void;
  loadPipelineConfig: (config: PipelineConfig) => void;
  createNewPipeline: () => void;
  saveCurrentPipeline: () => Promise<void>;
  validateCurrentPipeline: () => Promise<boolean>;

  // Validation state
  validationErrors: string[];
  validationWarnings: any[];

  // Execution state
  isExecuting: boolean;
  activeExecutionId: string | null;
  executionRecord: PipelineExecutionRecord | null;
  nodeStatuses: Record<string, NodeExecutionStatus>;
  executePipeline: (query?: string) => Promise<void>;
  handleExecutionEvent: (event: ExecutionEvent) => void;

  // UI Modals / Toggles
  isTemplateModalOpen: boolean;
  setTemplateModalOpen: (open: boolean) => void;
  isExecutionPanelOpen: boolean;
  setExecutionPanelOpen: (open: boolean) => void;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  schemas: [],
  categories: [],
  isLoadingSchemas: false,

  loadSchemas: async () => {
    if (get().schemas.length > 0 || get().isLoadingSchemas) return;
    set({ isLoadingSchemas: true });
    try {
      const [schemas, categories] = await Promise.all([
        fetchNodeSchemas(),
        fetchCategories(),
      ]);
      set({ schemas, categories, isLoadingSchemas: false });

      // Automatically load the first starter template if the canvas is empty
      if (get().nodes.length === 0) {
        try {
          const { fetchTemplates } = await import('../services/api');
          const templates = await fetchTemplates();
          if (templates && templates.length > 0) {
            get().loadPipelineConfig(templates[0].pipeline);
          }
        } catch (tErr) {
          console.warn('Could not auto-load starter template', tErr);
        }
      }
    } catch (e) {
      console.error('Failed to load schemas', e);
      set({ isLoadingSchemas: false });
    }
  },

  pipelineId: 'p-' + Math.random().toString(36).substring(2, 9),
  pipelineName: 'New RAG Pipeline',
  pipelineDescription: '',
  tags: ['draft'],
  nodes: [],
  edges: [],
  selectedNodeId: null,

  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },

  onEdgesChange: (changes) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },

  onConnect: (connection) => {
    set({
      edges: addEdge(
        {
          ...connection,
          animated: true,
          style: { stroke: '#22c55e', strokeWidth: 2 },
        },
        get().edges
      ),
    });
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  addNode: (type, position) => {
    const schema = get().schemas.find((s) => s.type === type);
    if (!schema) return;

    // Apply defaults
    const defaultConfig: Record<string, any> = {};
    for (const f of schema.fields) {
      if (f.default !== undefined) {
        defaultConfig[f.name] = f.default;
      }
    }

    const newNodeId = `node_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const newNode: RAGFlowNode = {
      id: newNodeId,
      type: 'ragNode',
      position,
      data: {
        schema,
        config: defaultConfig,
        label: schema.metadata.name,
        executionStatus: 'idle',
      },
    };

    set({
      nodes: [...get().nodes, newNode],
      selectedNodeId: newNodeId,
    });
  },

  updateNodeConfig: (nodeId, key, value) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id === nodeId) {
          return {
            ...n,
            data: {
              ...n.data,
              config: {
                ...n.data.config,
                [key]: value,
              },
            },
          };
        }
        return n;
      }),
    });
  },

  updateNodeLabel: (nodeId, label) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id === nodeId) {
          return {
            ...n,
            data: {
              ...n.data,
              label,
            },
          };
        }
        return n;
      }),
    });
  },

  removeNode: (nodeId) => {
    set({
      nodes: get().nodes.filter((n) => n.id !== nodeId),
      edges: get().edges.filter(
        (e) => e.source !== nodeId && e.target !== nodeId
      ),
      selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
    });
  },

  setPipelineName: (name) => set({ pipelineName: name }),
  setPipelineDescription: (desc) => set({ pipelineDescription: desc }),

  loadPipelineConfig: (config) => {
    const schemas = get().schemas;
    const reactFlowNodes: RAGFlowNode[] = config.nodes.map((n) => {
      const schema = schemas.find((s) => s.type === n.type) || {
        type: n.type,
        category: 'utility',
        metadata: {
          name: n.label || n.type,
          description: '',
          category: 'utility',
          icon: 'Box',
          tags: [],
        },
        input_ports: [],
        output_ports: [],
        fields: [],
      };

      return {
        id: n.id,
        type: 'ragNode',
        position: n.position || { x: 100, y: 100 },
        data: {
          schema,
          config: n.config || {},
          label: n.label || schema.metadata.name,
          executionStatus: 'idle',
        },
      };
    });

    const reactFlowEdges: Edge[] = config.edges.map((e) => ({
      id: e.id,
      source: e.source_node_id,
      sourceHandle: e.source_port,
      target: e.target_node_id,
      targetHandle: e.target_port,
      animated: true,
      style: { stroke: '#22c55e', strokeWidth: 2 },
    }));

    set({
      pipelineId: config.id,
      pipelineName: config.name,
      pipelineDescription: config.description || '',
      tags: config.tags || [],
      nodes: reactFlowNodes,
      edges: reactFlowEdges,
      selectedNodeId: null,
      validationErrors: [],
      validationWarnings: [],
    });
  },

  createNewPipeline: () => {
    set({
      pipelineId: 'p-' + Math.random().toString(36).substring(2, 9),
      pipelineName: 'Untitled RAG Pipeline',
      pipelineDescription: '',
      tags: ['custom'],
      nodes: [],
      edges: [],
      selectedNodeId: null,
      validationErrors: [],
      validationWarnings: [],
      executionRecord: null,
      nodeStatuses: {},
    });
  },

  saveCurrentPipeline: async () => {
    const state = get();
    const config: PipelineConfig = {
      id: state.pipelineId,
      name: state.pipelineName,
      description: state.pipelineDescription,
      tags: state.tags,
      nodes: state.nodes.map((n) => ({
        id: n.id,
        type: n.data.schema.type,
        label: n.data.label,
        position: n.position,
        config: n.data.config,
      })),
      edges: state.edges.map((e) => ({
        id: e.id,
        source_node_id: e.source,
        source_port: e.sourceHandle || 'output',
        target_node_id: e.target,
        target_port: e.targetHandle || 'input',
      })),
    };

    await savePipeline(config);
  },

  validateCurrentPipeline: async () => {
    const state = get();
    const config: PipelineConfig = {
      id: state.pipelineId,
      name: state.pipelineName,
      description: state.pipelineDescription,
      tags: state.tags,
      nodes: state.nodes.map((n) => ({
        id: n.id,
        type: n.data.schema.type,
        label: n.data.label,
        position: n.position,
        config: n.data.config,
      })),
      edges: state.edges.map((e) => ({
        id: e.id,
        source_node_id: e.source,
        source_port: e.sourceHandle || 'output',
        target_node_id: e.target,
        target_port: e.targetHandle || 'input',
      })),
    };

    try {
      const res = await validatePipeline(config);
      set({
        validationErrors: res.errors,
        validationWarnings: res.warnings,
      });
      return res.is_valid;
    } catch (e: any) {
      set({ validationErrors: [e.message] });
      return false;
    }
  },

  validationErrors: [],
  validationWarnings: [],

  isExecuting: false,
  activeExecutionId: null,
  executionRecord: null,
  nodeStatuses: {},

  executePipeline: async (query?: string) => {
    const state = get();
    set({
      isExecuting: true,
      isExecutionPanelOpen: true,
      nodeStatuses: {},
      nodes: state.nodes.map((n) => ({
        ...n,
        data: { ...n.data, executionStatus: 'idle' },
      })),
    });

    const config: PipelineConfig = {
      id: state.pipelineId,
      name: state.pipelineName,
      description: state.pipelineDescription,
      tags: state.tags,
      nodes: state.nodes.map((n) => ({
        id: n.id,
        type: n.data.schema.type,
        label: n.data.label,
        position: n.position,
        config: n.data.config,
      })),
      edges: state.edges.map((e) => ({
        id: e.id,
        source_node_id: e.source,
        source_port: e.sourceHandle || 'output',
        target_node_id: e.target,
        target_port: e.targetHandle || 'input',
      })),
    };

    const inputs: Record<string, any> = {};
    if (query) {
      inputs.query = query;
    }

    try {
      const record = await executePipelineDirect(config, inputs);
      set({
        executionRecord: record,
        isExecuting: false,
        nodes: get().nodes.map((n) => {
          const nr = record.node_records[n.id];
          return {
            ...n,
            data: {
              ...n.data,
              executionStatus: (nr?.status as NodeExecutionStatus) || 'completed',
            },
          };
        }),
      });
    } catch (e: any) {
      console.error('Execution error', e);
      set({ isExecuting: false });
    }
  },

  handleExecutionEvent: (event) => {
    if (event.node_id) {
      let status: NodeExecutionStatus = 'running';
      if (event.type === 'node_completed') status = 'completed';
      if (event.type === 'node_failed') status = 'failed';
      if (event.type === 'node_skipped') status = 'skipped';

      set({
        nodeStatuses: {
          ...get().nodeStatuses,
          [event.node_id]: status,
        },
        nodes: get().nodes.map((n) => {
          if (n.id === event.node_id) {
            return {
              ...n,
              data: {
                ...n.data,
                executionStatus: status,
              },
            };
          }
          return n;
        }),
      });
    }
  },

  isTemplateModalOpen: false,
  setTemplateModalOpen: (open) => set({ isTemplateModalOpen: open }),

  isExecutionPanelOpen: false,
  setExecutionPanelOpen: (open) => set({ isExecutionPanelOpen: open }),
}));
