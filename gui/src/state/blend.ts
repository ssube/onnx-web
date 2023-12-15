import {
  BlendParams,
  BrushParams,
  ModelParams,
  UpscaleParams,
} from '../types/params.js';
import { TabState } from './types.js';

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
