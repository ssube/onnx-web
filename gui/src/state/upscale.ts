import {
  HighresParams,
  ModelParams,
  UpscaleParams,
  UpscaleReqParams,
} from '../types/params.js';
import { TabState } from './types.js';

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
