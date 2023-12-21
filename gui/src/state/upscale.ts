import {
  BaseImgParams,
  HighresParams,
  ModelParams,
  UpscaleParams,
  UpscaleReqParams,
} from '../types/params.js';
import { Slice, TabState } from './types.js';

export interface UpscaleSlice {
  upscale: TabState<UpscaleReqParams>;
  upscaleHighres: HighresParams;
  upscaleModel: ModelParams;
  upscaleUpscale: UpscaleParams;

  resetUpscale(): void;

  setUpscale(params: Partial<UpscaleReqParams>): void;
  setUpscaleHighres(params: Partial<HighresParams>): void;
  setUpscaleModel(params: Partial<ModelParams>): void;
  setUpscaleUpscale(params: Partial<UpscaleParams>): void;
}

export function createUpscaleSlice<TState extends UpscaleSlice>(
  defaultParams: Required<BaseImgParams>,
  defaultHighres: HighresParams,
  defaultModel: ModelParams,
  defaultUpscale: UpscaleParams,
): Slice<TState, UpscaleSlice> {
  return (set) => ({
    upscale: {
      ...defaultParams,
      // eslint-disable-next-line no-null/no-null
      source: null,
    },
    upscaleHighres: {
      ...defaultHighres,
    },
    upscaleModel: {
      ...defaultModel,
    },
    upscaleUpscale: {
      ...defaultUpscale,
    },
    resetUpscale() {
      set({
        upscale: {
          ...defaultParams,
          // eslint-disable-next-line no-null/no-null
          source: null,
        },
      } as Partial<TState>);
    },
    setUpscale(source) {
      set((prev) => ({
        upscale: {
          ...prev.upscale,
          ...source,
        },
      } as Partial<TState>));
    },
    setUpscaleHighres(params) {
      set((prev) => ({
        upscaleHighres: {
          ...prev.upscaleHighres,
          ...params,
        },
      } as Partial<TState>));
    },
    setUpscaleModel(params) {
      set((prev) => ({
        upscaleModel: {
          ...prev.upscaleModel,
          ...params,
        },
      } as Partial<TState>));
    },
    setUpscaleUpscale(params) {
      set((prev) => ({
        upscaleUpscale: {
          ...prev.upscaleUpscale,
          ...params,
        },
      } as Partial<TState>));
    },
  });
}
