import { Slice } from './types.js';

export type Layout = 'horizontal' | 'vertical';

export const DEFAULT_SETTINGS = {
  historyWidth: 4,
  layout: 'vertical' as Layout,
} as const;

export interface SettingsSlice {
  historyWidth: number;
  layout: Layout;

  setHistoryWidth(width: number): void;
  setLayout(layout: Layout): void;
}

export function createSettingsSlice<TState extends SettingsSlice>(): Slice<TState, SettingsSlice> {
  return (set) => ({
    ...DEFAULT_SETTINGS,
    setLayout(layout) {
      set((prev) => ({
        ...prev,
        layout,
      }));
    },
    setHistoryWidth(historyWidth) {
      set((prev) => ({
        ...prev,
        historyWidth,
      }));
    },
  });
}
