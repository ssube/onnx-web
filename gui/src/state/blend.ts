import {
  BlendParams,
  BrushParams,
  ModelParams,
  UpscaleParams,
} from '../types/params.js';
import { DEFAULT_BRUSH, Slice, TabState } from './types.js';

export interface BlendSlice {
  blend: TabState<BlendParams>;
  blendBrush: BrushParams;
  blendModel: ModelParams;
  blendUpscale: UpscaleParams;

  resetBlend(): void;

  setBlend(blend: Partial<BlendParams>): void;
  setBlendBrush(brush: Partial<BrushParams>): void;
  setBlendModel(model: Partial<ModelParams>): void;
  setBlendUpscale(params: Partial<UpscaleParams>): void;
}

export function createBlendSlice<TState extends BlendSlice>(
  defaultModel: ModelParams,
  defaultUpscale: UpscaleParams,
): Slice<TState, BlendSlice> {
  return (set) => ({
    blend: {
      // eslint-disable-next-line no-null/no-null
      mask: null,
      sources: [],
    },
    blendBrush: {
      ...DEFAULT_BRUSH,
    },
    blendModel: {
      ...defaultModel,
    },
    blendUpscale: {
      ...defaultUpscale,
    },
    resetBlend() {
      set((prev) => ({
        blend: {
          // eslint-disable-next-line no-null/no-null
          mask: null,
          sources: [] as Array<Blob>,
        },
      } as Partial<TState>));
    },
    setBlend(blend) {
      set((prev) => ({
        blend: {
          ...prev.blend,
          ...blend,
        },
      } as Partial<TState>));
    },
    setBlendBrush(brush) {
      set((prev) => ({
        blendBrush: {
          ...prev.blendBrush,
          ...brush,
        },
      } as Partial<TState>));
    },
    setBlendModel(model) {
      set((prev) => ({
        blendModel: {
          ...prev.blendModel,
          ...model,
        },
      } as Partial<TState>));
    },
    setBlendUpscale(params) {
      set((prev) => ({
        blendUpscale: {
          ...prev.blendUpscale,
          ...params,
        },
      } as Partial<TState>));
    },
  });
}
