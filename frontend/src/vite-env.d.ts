/// <reference types="vite/client" />

declare module 'lucide-react' {
  import { FC, SVGProps } from 'react';
  export interface LucideProps extends SVGProps<SVGSVGElement> {
    size?: string | number;
    color?: string;
    strokeWidth?: string | number;
    className?: string;
  }
  export type Icon = FC<LucideProps>;
  export const AlertCircle: Icon;
  export const AlertTriangle: Icon;
  export const ArrowDownRight: Icon;
  export const ArrowRight: Icon;
  export const BookOpen: Icon;
  export const Check: Icon;
  export const CheckCircle: Icon;
  export const CheckCircle2: Icon;
  export const ChevronDown: Icon;
  export const ChevronUp: Icon;
  export const Clock: Icon;
  export const Copy: Icon;
  export const Cpu: Icon;
  export const Database: Icon;
  export const FileText: Icon;
  export const Filter: Icon;
  export const FolderOpen: Icon;
  export const Layers: Icon;
  export const Loader2: Icon;
  export const MessageSquare: Icon;
  export const Play: Icon;
  export const Plus: Icon;
  export const Save: Icon;
  export const Scissors: Icon;
  export const Search: Icon;
  export const Send: Icon;
  export const ShieldCheck: Icon;
  export const Sparkles: Icon;
  export const Terminal: Icon;
  export const Trash2: Icon;
  export const Upload: Icon;
  export const X: Icon;
  const icons: Record<string, Icon>;
  export default icons;
}
