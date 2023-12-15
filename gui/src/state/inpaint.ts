import {
  BrushParams,
  HighresParams,
  InpaintParams,
  ModelParams,
  OutpaintPixels,
  UpscaleParams,
} from '../types/params.js';
import { TabState } from './types.js';

export interface InpaintSlice {
  inpaint: TabState<InpaintParams>;
  inpaintBrush: BrushParams;
  inpaintModel: ModelParams;
  inpaintHighres: HighresParams;
  inpaintUpscale: UpscaleParams;
  outpaint: OutpaintPixels;

  resetInpaint(): void;

  setInpaint(params: Partial<InpaintParams>): void;
  setInpaintBrush(brush: Partial<BrushParams>): void;
  setInpaintModel(params: Partial<ModelParams>): void;
  setInpaintHighres(params: Partial<HighresParams>): void;
  setInpaintUpscale(params: Partial<UpscaleParams>): void;
  setOutpaint(pixels: Partial<OutpaintPixels>): void;
}
