import React, { memo } from 'react';
import { Handle, NodeProps, Position } from '@xyflow/react';
import {
  AlertCircle,
  ArrowDownRight,
  BookOpen,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  Filter,
  Layers,
  Loader2,
  MessageSquare,
  Scissors,
  Sparkles,
  Terminal,
} from 'lucide-react';
import { RAGFlowNodeData } from '../stores/pipelineStore';

const CATEGORY_COLORS: Record<string, { border: string; bg: string; text: string; handle: string }> = {
  input: { border: 'border-blue-500/50', bg: 'bg-blue-950/40', text: 'text-blue-400', handle: '!bg-blue-400' },
  output: { border: 'border-emerald-500/50', bg: 'bg-emerald-950/40', text: 'text-emerald-400', handle: '!bg-emerald-400' },
  loader: { border: 'border-amber-500/50', bg: 'bg-amber-950/40', text: 'text-amber-400', handle: '!bg-amber-400' },
  chunker: { border: 'border-orange-500/50', bg: 'bg-orange-950/40', text: 'text-orange-400', handle: '!bg-orange-400' },
  embedder: { border: 'border-cyan-500/50', bg: 'bg-cyan-950/40', text: 'text-cyan-400', handle: '!bg-cyan-400' },
  vector_store: { border: 'border-indigo-500/50', bg: 'bg-indigo-950/40', text: 'text-indigo-400', handle: '!bg-indigo-400' },
  retriever: { border: 'border-violet-500/50', bg: 'bg-violet-950/40', text: 'text-violet-400', handle: '!bg-violet-400' },
  reranker: { border: 'border-fuchsia-500/50', bg: 'bg-fuchsia-950/40', text: 'text-fuchsia-400', handle: '!bg-fuchsia-400' },
  generator: { border: 'border-rose-500/50', bg: 'bg-rose-950/40', text: 'text-rose-400', handle: '!bg-rose-400' },
  prompt: { border: 'border-teal-500/50', bg: 'bg-teal-950/40', text: 'text-teal-400', handle: '!bg-teal-400' },
  utility: { border: 'border-slate-500/50', bg: 'bg-slate-900/60', text: 'text-slate-400', handle: '!bg-slate-400' },
};

function getCategoryIcon(cat: string) {
  switch (cat) {
    case 'input': return MessageSquare;
    case 'output': return Terminal;
    case 'loader': return FileText;
    case 'chunker': return Scissors;
    case 'embedder': return Layers;
    case 'vector_store': return Database;
    case 'retriever': return ArrowDownRight;
    case 'reranker': return Filter;
    case 'generator': return Sparkles;
    case 'prompt': return BookOpen;
    default: return Cpu;
  }
}

export const CustomNode = memo(({ id, data, selected }: NodeProps<any>) => {
  const nodeData = data as RAGFlowNodeData;
  const schema = nodeData.schema;
  const category = schema?.category || 'utility';
  const color = CATEGORY_COLORS[category] || CATEGORY_COLORS.utility;
  const IconComponent = getCategoryIcon(category);
  const status = nodeData.executionStatus || 'idle';

  return (
    <div
      className={`relative min-w-[240px] max-w-[320px] rounded-xl border bg-slate-900/95 backdrop-blur-md shadow-2xl transition-all duration-200 ${
        selected ? 'ring-2 ring-brand-500 border-transparent shadow-brand-500/10' : color.border
      }`}
    >
      {/* Node Header */}
      <div className={`flex items-center justify-between px-3.5 py-2.5 rounded-t-xl border-b border-slate-800/80 ${color.bg}`}>
        <div className="flex items-center gap-2 overflow-hidden">
          <div className={`p-1.5 rounded-lg bg-slate-900/80 ${color.text}`}>
            <IconComponent className="w-4 h-4" />
          </div>
          <div className="flex flex-col truncate">
            <span className="text-xs font-semibold text-slate-100 truncate">
              {nodeData.label || schema?.metadata?.name || id}
            </span>
            <span className="text-[10px] text-slate-400 font-mono lowercase truncate">
              {schema?.type}
            </span>
          </div>
        </div>

        {/* Execution Status Badge */}
        <div className="flex items-center pl-2">
          {status === 'running' && (
            <span className="flex items-center gap-1 text-[10px] font-medium text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded-full border border-amber-500/30">
              <Loader2 className="w-3 h-3 animate-spin" />
              Run
            </span>
          )}
          {status === 'completed' && (
            <span className="flex items-center gap-1 text-[10px] font-medium text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-full border border-emerald-500/30">
              <CheckCircle2 className="w-3 h-3" />
              Done
            </span>
          )}
          {status === 'failed' && (
            <span className="flex items-center gap-1 text-[10px] font-medium text-rose-400 bg-rose-950/60 px-2 py-0.5 rounded-full border border-rose-500/30">
              <AlertCircle className="w-3 h-3" />
              Fail
            </span>
          )}
        </div>
      </div>

      {/* Ports Area */}
      <div className="py-2.5 px-1 space-y-2">
        {/* Input Ports (Left) */}
        {schema?.input_ports?.map((port) => (
          <div key={`in-${port.name}`} className="relative flex items-center h-6 pl-3">
            <Handle
              type="target"
              position={Position.Left}
              id={port.name}
              className={`w-2.5 h-2.5 -left-1.5 transition-transform hover:scale-125 ${color.handle}`}
            />
            <div className="flex items-center gap-1.5 text-[11px] text-slate-300">
              <span className="font-mono text-slate-400 text-[10px]">{port.name}</span>
              {port.required && <span className="text-rose-400 text-[10px]">*</span>}
            </div>
          </div>
        ))}

        {/* Output Ports (Right) */}
        {schema?.output_ports?.map((port) => (
          <div key={`out-${port.name}`} className="relative flex items-center justify-end h-6 pr-3">
            <span className="font-mono text-slate-300 text-[11px] text-right mr-1.5">
              {port.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={port.name}
              className={`w-2.5 h-2.5 -right-1.5 transition-transform hover:scale-125 ${color.handle}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
});

CustomNode.displayName = 'CustomNode';
