import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle,
  FolderOpen,
  Plus,
  Save,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';

export const Toolbar: React.FC = () => {
  const {
    pipelineName,
    setPipelineName,
    saveCurrentPipeline,
    createNewPipeline,
    validateCurrentPipeline,
    validationErrors,
    validationWarnings,
    setTemplateModalOpen,
  } = usePipelineStore();

  const [isSaved, setIsSaved] = useState(false);
  const [isValidating, setIsValidating] = useState(false);

  const handleSave = async () => {
    try {
      await saveCurrentPipeline();
      setIsSaved(true);
      setTimeout(() => setIsSaved(false), 2000);
    } catch (e: any) {
      alert('Save failed: ' + e.message);
    }
  };

  const handleValidate = async () => {
    setIsValidating(true);
    await validateCurrentPipeline();
    setIsValidating(false);
  };

  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between select-none z-30">
      {/* Brand & Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-600 to-emerald-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Sparkles className="w-4 h-4 text-slate-950 stroke-[2.5]" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-100 tracking-tight">
              RAGForge
            </h1>
            <span className="text-[10px] text-brand-400 font-medium">
              v0.1.0 Visual Builder
            </span>
          </div>
        </div>

        <div className="h-5 w-px bg-slate-800" />

        {/* Pipeline Name input */}
        <input
          type="text"
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          className="text-xs font-semibold text-slate-200 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 focus:border-brand-500 focus:outline-none transition-colors w-64"
        />
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-2">
        {/* Validation Status Indicator */}
        {validationErrors.length > 0 ? (
          <div className="flex items-center gap-1.5 text-xs text-rose-400 bg-rose-950/40 border border-rose-500/30 px-2.5 py-1 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{validationErrors.length} Issue(s)</span>
          </div>
        ) : validationWarnings.length > 0 ? (
          <div className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-950/40 border border-amber-500/30 px-2.5 py-1 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{validationWarnings.length} Warning(s)</span>
          </div>
        ) : null}

        <button
          onClick={() => setTemplateModalOpen(true)}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-slate-100 bg-slate-800/80 hover:bg-slate-800 px-3 py-1.5 rounded-lg transition-colors border border-slate-700/60 font-medium"
        >
          <FolderOpen className="w-3.5 h-3.5 text-brand-400" />
          <span>Templates</span>
        </button>

        <button
          onClick={createNewPipeline}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-slate-100 bg-slate-800/80 hover:bg-slate-800 px-3 py-1.5 rounded-lg transition-colors border border-slate-700/60 font-medium"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New</span>
        </button>

        <button
          onClick={handleValidate}
          disabled={isValidating}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-slate-100 bg-slate-800/80 hover:bg-slate-800 px-3 py-1.5 rounded-lg transition-colors border border-slate-700/60 font-medium"
        >
          <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
          <span>Validate</span>
        </button>

        <button
          onClick={handleSave}
          className={`flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all shadow-sm ${
            isSaved
              ? 'bg-emerald-600 text-white'
              : 'bg-brand-600 hover:bg-brand-500 text-white'
          }`}
        >
          {isSaved ? (
            <>
              <Check className="w-3.5 h-3.5" />
              <span>Saved!</span>
            </>
          ) : (
            <>
              <Save className="w-3.5 h-3.5" />
              <span>Save</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
