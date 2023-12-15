import {
  BaseImgParams,
} from '../types/params.js';
import { TabState, Theme } from './types.js';

export interface DefaultSlice {
  defaults: TabState<BaseImgParams>;
  theme: Theme;

  setDefaults(param: Partial<BaseImgParams>): void;
  setTheme(theme: Theme): void;
}
