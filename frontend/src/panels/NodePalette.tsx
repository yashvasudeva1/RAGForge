import React, { useState } from 'react';
import {
  ArrowDownRight,
  BookOpen,
  Cpu,
  Database,
  FileText,
  Filter,
  Layers,
  MessageSquare,
  Scissors,
  Search,
  Sparkles,
  Terminal,
} from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';
import { NodeSchema } from '../types';

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

export const NodePalette: React.FC = () => {
  const { schemas, categories, addNode } = usePipelineStore();
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const filteredSchemas = schemas.filter((schema) => {
    const matchesSearch =
      schema.metadata?.name.toLowerCase().includes(search.toLowerCase()) ||
      schema.type.toLowerCase().includes(search.toLowerCase()) ||
      schema.metadata?.description?.toLowerCase().includes(search.toLowerCase());
    const matchesCategory =
      !selectedCategory || schema.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/ragforge-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="w-72 h-full flex flex-col bg-slate-900 border-r border-slate-800 z-10 select-none">
      {/* Search Header */}
      <div className="p-3 border-b border-slate-800/80">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
          <input
            type="text"
            placeholder="Search 49 components..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-950 text-slate-200 text-xs pl-9 pr-3 py-2 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500 transition-colors"
          />
        </div>

        {/* Category Pills */}
        <div className="flex gap-1.5 overflow-x-auto py-2 no-scrollbar">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-2 py-1 rounded-md text-[10px] font-medium whitespace-nowrap transition-colors ${
              selectedCategory === null
                ? 'bg-brand-500/20 text-brand-400 border border-brand-500/40'
                : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({schemas.length})
          </button>
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() =>
                setSelectedCategory(selectedCategory === cat.id ? null : cat.id)
              }
              className={`px-2 py-1 rounded-md text-[10px] font-medium whitespace-nowrap transition-colors ${
                selectedCategory === cat.id
                  ? 'bg-brand-500/20 text-brand-400 border border-brand-500/40'
                  : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
              }`}
            >
              {cat.label} ({cat.node_count})
            </button>
          ))}
        </div>
      </div>

      {/* Node Cards List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredSchemas.map((schema) => {
          const Icon = getCategoryIcon(schema.category);
          return (
            <div
              key={schema.type}
              draggable
              onDragStart={(e) => onDragStart(e, schema.type)}
              onClick={() => {
                // Click to add at center
                addNode(schema.type, {
                  x: 350 + Math.random() * 100,
                  y: 150 + Math.random() * 100,
                });
              }}
              className="group p-2.5 rounded-lg border border-slate-800/80 bg-slate-950/60 hover:bg-slate-800/50 hover:border-slate-700/80 cursor-grab active:cursor-grabbing transition-all"
            >
              <div className="flex items-start gap-2.5">
                <div className="p-1.5 rounded-md bg-slate-900 border border-slate-800 text-brand-400 group-hover:border-brand-500/40 transition-colors">
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-slate-200 group-hover:text-brand-400 transition-colors truncate">
                      {schema.metadata?.name || schema.type}
                    </h4>
                  </div>
                  <p className="text-[10px] text-slate-400 line-clamp-2 mt-0.5 leading-relaxed">
                    {schema.metadata?.description || 'Pipeline node component'}
                  </p>
                </div>
              </div>
            </div>
          );
        })}

        {filteredSchemas.length === 0 && (
          <div className="text-center py-8 text-slate-500 text-xs">
            No components match your search.
          </div>
        )}
      </div>
    </aside>
  );
};
