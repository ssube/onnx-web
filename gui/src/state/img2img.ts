
import {
  HighresParams,
  Img2ImgParams,
  ModelParams,
  UpscaleParams,
} from '../types/params.js';
import { TabState } from './types.js';

export interface Img2ImgSlice {
  img2img: TabState<Img2ImgParams>;
  img2imgModel: ModelParams;
  img2imgHighres: HighresParams;
  img2imgUpscale: UpscaleParams;

  resetImg2Img(): void;

  setImg2Img(params: Partial<Img2ImgParams>): void;
  setImg2ImgModel(params: Partial<ModelParams>): void;
  setImg2ImgHighres(params: Partial<HighresParams>): void;
  setImg2ImgUpscale(params: Partial<UpscaleParams>): void;
}
