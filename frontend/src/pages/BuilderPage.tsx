import React, { useEffect } from 'react';
import { Canvas } from '../canvas/Canvas';
import { NodePalette } from '../panels/NodePalette';
import { ConfigPanel } from '../panels/ConfigPanel';
import { ExecutionPanel } from '../panels/ExecutionPanel';
import { Toolbar } from '../panels/Toolbar';
import { TemplateGalleryModal } from './TemplateGalleryModal';
import { usePipelineStore } from '../stores/pipelineStore';

export const BuilderPage: React.FC = () => {
  const { loadSchemas } = usePipelineStore();

  useEffect(() => {
    loadSchemas();
  }, [loadSchemas]);

  return (
    <div className="w-screen h-screen flex flex-col bg-slate-950 overflow-hidden">
      {/* Top Navigation */}
      <Toolbar />

      {/* Main Workspace */}
      <div className="flex-1 flex relative min-h-0">
        {/* Left Component Palette */}
        <NodePalette />

        {/* Center Flow Canvas */}
        <main className="flex-1 h-full relative">
          <Canvas />
          <ExecutionPanel />
        </main>

        {/* Right Config Inspector */}
        <ConfigPanel />
      </div>

      {/* Template Browser Modal */}
      <TemplateGalleryModal />
    </div>
  );
};
