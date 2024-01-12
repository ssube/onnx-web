import { Slice } from './types.js';

export type Layout = 'horizontal' | 'vertical';

export const DEFAULT_LAYOUT = {
  historyWidth: 4,
  layout: 'vertical' as Layout,
} as const;

export interface SettingsSlice {
  historyWidth: number;
  layout: Layout;

  setLayout(layout: Layout): void;
  setWidth(width: number): void;
}

export function createSettingsSlice<TState extends SettingsSlice>(): Slice<TState, SettingsSlice> {
  return (set) => ({
    ...DEFAULT_LAYOUT,
    setLayout(layout) {
      set((prev) => ({
        ...prev,
        layout,
      }));
    },
    setWidth(width) {
      set((prev) => ({
        ...prev,
        historyWidth: width,
      }));
    },
  });
}
