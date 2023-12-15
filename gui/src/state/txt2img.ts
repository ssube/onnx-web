import { PipelineGrid } from '../client/utils.js';
import {
  HighresParams,
  ModelParams,
  Txt2ImgParams,
  UpscaleParams,
} from '../types/params.js';
import { TabState } from './types.js';

export interface Txt2ImgSlice {
  txt2img: TabState<Txt2ImgParams>;
  txt2imgModel: ModelParams;
  txt2imgHighres: HighresParams;
  txt2imgUpscale: UpscaleParams;
  txt2imgVariable: PipelineGrid;

  resetTxt2Img(): void;

  setTxt2Img(params: Partial<Txt2ImgParams>): void;
  setTxt2ImgModel(params: Partial<ModelParams>): void;
  setTxt2ImgHighres(params: Partial<HighresParams>): void;
  setTxt2ImgUpscale(params: Partial<UpscaleParams>): void;
  setTxt2ImgVariable(params: Partial<PipelineGrid>): void;
}
