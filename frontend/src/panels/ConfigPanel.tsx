import React from 'react';
import { Trash2, Upload, X } from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';
import { NodeField } from '../types';
import { uploadDocument } from '../services/api';

export const ConfigPanel: React.FC = () => {
  const {
    nodes,
    selectedNodeId,
    setSelectedNodeId,
    updateNodeConfig,
    updateNodeLabel,
    removeNode,
  } = usePipelineStore();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <aside className="w-80 h-full bg-slate-900 border-l border-slate-800 p-6 flex flex-col items-center justify-center text-center text-slate-500 text-xs select-none">
        <div className="w-12 h-12 rounded-full border border-dashed border-slate-700 flex items-center justify-center mb-3 text-slate-600">
          ⚙️
        </div>
        <p className="font-medium text-slate-400">No Node Selected</p>
        <p className="mt-1 max-w-[200px] text-slate-500">
          Click any component on the canvas to configure its settings and parameters.
        </p>
      </aside>
    );
  }

  const { schema, config, label } = selectedNode.data;
  const fields: NodeField[] = schema?.fields || [];

  const handleFileUpload = async (
    fieldName: string,
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await uploadDocument(file);
      updateNodeConfig(selectedNode.id, fieldName, res.file_path);
    } catch (err) {
      alert('Upload failed: ' + err);
    }
  };

  return (
    <aside className="w-80 h-full bg-slate-900 border-l border-slate-800 flex flex-col z-10 select-none">
      {/* Header */}
      <div className="p-3.5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex-1 min-w-0 pr-2">
          <input
            type="text"
            value={label || schema?.metadata?.name || ''}
            onChange={(e) => updateNodeLabel(selectedNode.id, e.target.value)}
            className="w-full text-xs font-semibold text-slate-100 bg-transparent border-b border-transparent hover:border-slate-700 focus:border-brand-500 focus:outline-none py-0.5 truncate"
          />
          <div className="text-[10px] text-slate-400 font-mono">
            {schema?.type}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => removeNode(selectedNode.id)}
            title="Delete node"
            className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-950/40 rounded-md transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setSelectedNodeId(null)}
            title="Close panel"
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-md transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Description */}
      {schema?.metadata?.description && (
        <div className="p-3 bg-slate-950/40 border-b border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
          {schema.metadata.description}
        </div>
      )}

      {/* Form Fields */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {fields.map((field) => {
          const value =
            config[field.name] !== undefined ? config[field.name] : field.default;

          return (
            <div key={field.name} className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-slate-300">
                  {field.label}
                  {field.required && (
                    <span className="text-rose-400 ml-0.5">*</span>
                  )}
                </label>
                {field.field_type === 'file_path' && (
                  <label className="flex items-center gap-1 text-[10px] text-brand-400 hover:text-brand-300 cursor-pointer">
                    <Upload className="w-3 h-3" />
                    <span>Upload</span>
                    <input
                      type="file"
                      className="hidden"
                      onChange={(e) => handleFileUpload(field.name, e)}
                    />
                  </label>
                )}
              </div>

              {/* Render by field type */}
              {field.field_type === 'boolean' ? (
                <div className="flex items-center gap-2 pt-1">
                  <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(e) =>
                      updateNodeConfig(
                        selectedNode.id,
                        field.name,
                        e.target.checked
                      )
                    }
                    className="w-4 h-4 rounded border-slate-700 bg-slate-950 text-brand-500 focus:ring-brand-500"
                  />
                  <span className="text-xs text-slate-400">
                    {field.description || 'Enable option'}
                  </span>
                </div>
              ) : field.field_type === 'select' && field.options ? (
                <select
                  value={value}
                  onChange={(e) =>
                    updateNodeConfig(selectedNode.id, field.name, e.target.value)
                  }
                  className="w-full bg-slate-950 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500"
                >
                  {field.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : field.field_type === 'textarea' ? (
                <textarea
                  rows={4}
                  value={value ?? ''}
                  onChange={(e) =>
                    updateNodeConfig(selectedNode.id, field.name, e.target.value)
                  }
                  placeholder={field.placeholder || ''}
                  className="w-full bg-slate-950 text-slate-200 text-xs p-2.5 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500 font-mono"
                />
              ) : field.field_type === 'integer' || field.field_type === 'float' ? (
                <input
                  type="number"
                  value={value ?? 0}
                  step={field.step || (field.field_type === 'float' ? 0.05 : 1)}
                  min={field.min_val}
                  max={field.max_val}
                  onChange={(e) =>
                    updateNodeConfig(
                      selectedNode.id,
                      field.name,
                      field.field_type === 'float'
                        ? parseFloat(e.target.value)
                        : parseInt(e.target.value, 10)
                    )
                  }
                  className="w-full bg-slate-950 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500 font-mono"
                />
              ) : (
                <input
                  type={field.field_type === 'secret' ? 'password' : 'text'}
                  value={value ?? ''}
                  placeholder={field.placeholder || ''}
                  onChange={(e) =>
                    updateNodeConfig(selectedNode.id, field.name, e.target.value)
                  }
                  className="w-full bg-slate-950 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-800 focus:outline-none focus:border-brand-500 font-mono"
                />
              )}

              {field.description && field.field_type !== 'boolean' && (
                <p className="text-[10px] text-slate-500 leading-tight">
                  {field.description}
                </p>
              )}
            </div>
          );
        })}

        {fields.length === 0 && (
          <p className="text-slate-500 text-xs italic text-center py-4">
            This component requires no configuration.
          </p>
        )}
      </div>
    </aside>
  );
};
