import React, { useEffect, useState } from 'react';
import { ArrowRight, BookOpen, Layers, Sparkles, X } from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';
import { fetchTemplates } from '../services/api';

export const TemplateGalleryModal: React.FC = () => {
  const { isTemplateModalOpen, setTemplateModalOpen, loadPipelineConfig } =
    usePipelineStore();
  const [templates, setTemplates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isTemplateModalOpen) {
      setLoading(true);
      fetchTemplates()
        .then((res) => setTemplates(res))
        .catch((err) => console.error('Failed to load templates', err))
        .finally(() => setLoading(false));
    }
  }, [isTemplateModalOpen]);

  if (!isTemplateModalOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-3xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-brand-500/10 text-brand-400 border border-brand-500/20">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                Starter RAG Templates
              </h2>
              <p className="text-xs text-slate-400">
                Select a pre-built architecture to instantly experiment on the canvas.
              </p>
            </div>
          </div>
          <button
            onClick={() => setTemplateModalOpen(false)}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-4">
          {loading ? (
            <div className="text-center py-12 text-slate-500 text-xs">
              Loading templates...
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {templates.map((tpl) => (
                <div
                  key={tpl.id}
                  className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-brand-500/40 hover:bg-slate-800/40 transition-all flex flex-col justify-between group"
                >
                  <div>
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-slate-100 group-hover:text-brand-400 transition-colors">
                        {tpl.name}
                      </h3>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
                        {tpl.node_count} nodes
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                      {tpl.description}
                    </p>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {tpl.tags?.map((tag: string) => (
                        <span
                          key={tag}
                          className="text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded-md"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={() => {
                      loadPipelineConfig(tpl.pipeline);
                      setTemplateModalOpen(false);
                    }}
                    className="mt-4 w-full flex items-center justify-center gap-1.5 text-xs font-semibold bg-brand-600 hover:bg-brand-500 text-white py-2 rounded-lg transition-colors"
                  >
                    <span>Load Pipeline</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
