import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Copy,
  Loader2,
  Play,
  Send,
} from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';

export const ExecutionPanel: React.FC = () => {
  const {
    isExecuting,
    executionRecord,
    executePipeline,
    isExecutionPanelOpen,
    setExecutionPanelOpen,
  } = usePipelineStore();

  const [query, setQuery] = useState('What is RAG and how does it work?');
  const [activeTab, setActiveTab] = useState<'answer' | 'nodes' | 'raw'>('answer');

  const handleRun = () => {
    executePipeline(query);
  };

  const finalOutput = executionRecord?.outputs;
  const answer =
    finalOutput?.generation_result?.answer ||
    finalOutput?.answer ||
    (typeof finalOutput === 'string' ? finalOutput : null);

  const sources = finalOutput?.generation_result?.source_chunks || [];

  return (
    <div
      className={`fixed bottom-0 left-72 right-80 bg-slate-900 border-t border-slate-800 z-20 transition-all duration-300 flex flex-col shadow-2xl ${
        isExecutionPanelOpen ? 'h-80' : 'h-11'
      }`}
    >
      {/* Top Bar / Controls */}
      <div className="h-11 px-4 border-b border-slate-800/80 flex items-center justify-between bg-slate-950/60">
        <div className="flex items-center gap-2 flex-1 max-w-xl">
          <input
            type="text"
            placeholder="Type a test question for your pipeline..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isExecuting && handleRun()}
            className="flex-1 bg-slate-900 text-xs text-slate-100 px-3 py-1.5 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500 font-sans"
          />
          <button
            onClick={handleRun}
            disabled={isExecuting}
            className="flex items-center gap-1.5 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-xs font-semibold px-3 py-1.5 rounded-lg shadow-sm transition-all"
          >
            {isExecuting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run</span>
              </>
            )}
          </button>
        </div>

        {/* Status & Toggle */}
        <div className="flex items-center gap-3">
          {executionRecord && (
            <div className="flex items-center gap-2 text-[11px] text-slate-400">
              {executionRecord.status === 'completed' ? (
                <span className="flex items-center gap-1 text-emerald-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Success
                </span>
              ) : (
                <span className="flex items-center gap-1 text-rose-400">
                  <AlertCircle className="w-3.5 h-3.5" />
                  Failed
                </span>
              )}
              {executionRecord.duration_ms && (
                <span className="flex items-center gap-1 font-mono text-slate-400">
                  <Clock className="w-3 h-3" />
                  {(executionRecord.duration_ms / 1000).toFixed(2)}s
                </span>
              )}
            </div>
          )}

          <button
            onClick={() => setExecutionPanelOpen(!isExecutionPanelOpen)}
            className="p-1 text-slate-400 hover:text-slate-200 rounded-md transition-colors"
          >
            {isExecutionPanelOpen ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronUp className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded Content Area */}
      {isExecutionPanelOpen && (
        <div className="flex-1 flex flex-col min-h-0 bg-slate-950/40">
          {/* Sub Tabs */}
          <div className="flex gap-4 px-4 pt-2 border-b border-slate-800/80 text-xs">
            <button
              onClick={() => setActiveTab('answer')}
              className={`pb-1.5 font-medium transition-colors border-b-2 ${
                activeTab === 'answer'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              Answer & Sources
            </button>
            <button
              onClick={() => setActiveTab('nodes')}
              className={`pb-1.5 font-medium transition-colors border-b-2 ${
                activeTab === 'nodes'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              Node Logs & Metrics
            </button>
            <button
              onClick={() => setActiveTab('raw')}
              className={`pb-1.5 font-medium transition-colors border-b-2 ${
                activeTab === 'raw'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              Raw Execution JSON
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto p-4 select-text">
            {activeTab === 'answer' && (
              <div className="space-y-4">
                {answer ? (
                  <div>
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      Answer
                    </h4>
                    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 text-sm text-slate-100 whitespace-pre-wrap leading-relaxed font-sans">
                      {answer}
                    </div>

                    {/* Sources Citations */}
                    {sources.length > 0 && (
                      <div className="mt-4">
                        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Sources & Retrieved Context ({sources.length})
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {sources.map((s: any, idx: number) => (
                            <div
                              key={idx}
                              className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/60 text-xs space-y-1"
                            >
                              <div className="flex items-center justify-between text-[11px] text-slate-400">
                                <span className="font-semibold text-brand-400">
                                  #{idx + 1} {s.source || 'Document'}
                                </span>
                                {s.score !== undefined && (
                                  <span className="font-mono text-slate-400">
                                    Score: {Number(s.score).toFixed(3)}
                                  </span>
                                )}
                              </div>
                              <p className="text-slate-300 text-[11px] line-clamp-3">
                                {s.content || s.chunk?.content}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : isExecuting ? (
                  <div className="flex items-center justify-center py-12 text-slate-400 gap-2 text-xs">
                    <Loader2 className="w-4 h-4 animate-spin text-brand-500" />
                    <span>Executing pipeline stages...</span>
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500 text-xs">
                    Run the pipeline above to see the generated answer and retrieved sources.
                  </div>
                )}
              </div>
            )}

            {activeTab === 'nodes' && (
              <div className="space-y-2">
                {executionRecord?.node_records &&
                  Object.values(executionRecord.node_records).map((record) => (
                    <div
                      key={record.node_id}
                      className="p-3 rounded-lg bg-slate-900/80 border border-slate-800/80 flex items-center justify-between text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-medium text-slate-200">
                          {record.node_type}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          ({record.node_id})
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        {record.duration_ms && (
                          <span className="font-mono text-[11px] text-slate-400">
                            {record.duration_ms.toFixed(1)}ms
                          </span>
                        )}
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            record.status === 'completed'
                              ? 'bg-emerald-950/60 text-emerald-400'
                              : record.status === 'failed'
                              ? 'bg-rose-950/60 text-rose-400'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {record.status}
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            )}

            {activeTab === 'raw' && (
              <pre className="text-xs font-mono text-slate-300 p-3 bg-slate-900 rounded-lg overflow-x-auto">
                {JSON.stringify(executionRecord, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
