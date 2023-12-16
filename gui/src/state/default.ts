import {
  BaseImgParams,
} from '../types/params.js';
import { Slice, TabState, Theme } from './types.js';

export interface DefaultSlice {
  defaults: TabState<BaseImgParams>;
  theme: Theme;

  setDefaults(param: Partial<BaseImgParams>): void;
  setTheme(theme: Theme): void;
}

export function createDefaultSlice<TState extends DefaultSlice>(defaultParams: Required<BaseImgParams>): Slice<TState, DefaultSlice> {
  return (set) => ({
    defaults: {
      ...defaultParams,
    },
    theme: '',
    setDefaults(params) {
      set((prev) => ({
        defaults: {
          ...prev.defaults,
          ...params,
        }
      } as Partial<TState>));
    },
    setTheme(theme) {
      set((prev) => ({
        theme,
      } as Partial<TState>));
    }
  });
}
