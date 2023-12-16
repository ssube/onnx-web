import { PipelineGrid } from '../client/utils.js';
import { ServerParams } from '../config.js';
import {
  BaseImgParams,
  HighresParams,
  ModelParams,
  Txt2ImgParams,
  UpscaleParams,
} from '../types/params.js';
import { Slice, TabState } from './types.js';

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

// eslint-disable-next-line max-params
export function createTxt2ImgSlice<TState extends Txt2ImgSlice>(
  server: ServerParams,
  defaultParams: Required<BaseImgParams>,
  defaultHighres: HighresParams,
  defaultModel: ModelParams,
  defaultUpscale: UpscaleParams,
  defaultGrid: PipelineGrid,
): Slice<TState, Txt2ImgSlice> {
  return (set) => ({
    txt2img: {
      ...defaultParams,
      width: server.width.default,
      height: server.height.default,
    },
    txt2imgHighres: {
      ...defaultHighres,
    },
    txt2imgModel: {
      ...defaultModel,
    },
    txt2imgUpscale: {
      ...defaultUpscale,
    },
    txt2imgVariable: {
      ...defaultGrid,
    },
    setTxt2Img(params) {
      set((prev) => ({
        txt2img: {
          ...prev.txt2img,
          ...params,
        },
      } as Partial<TState>));
    },
    setTxt2ImgHighres(params) {
      set((prev) => ({
        txt2imgHighres: {
          ...prev.txt2imgHighres,
          ...params,
        },
      } as Partial<TState>));
    },
    setTxt2ImgModel(params) {
      set((prev) => ({
        txt2imgModel: {
          ...prev.txt2imgModel,
          ...params,
        },
      } as Partial<TState>));
    },
    setTxt2ImgUpscale(params) {
      set((prev) => ({
        txt2imgUpscale: {
          ...prev.txt2imgUpscale,
          ...params,
        },
      } as Partial<TState>));
    },
    setTxt2ImgVariable(params) {
      set((prev) => ({
        txt2imgVariable: {
          ...prev.txt2imgVariable,
          ...params,
        },
      } as Partial<TState>));
    },
    resetTxt2Img() {
      set({
        txt2img: {
          ...defaultParams,
          width: server.width.default,
          height: server.height.default,
        },
      } as Partial<TState>);
    },
  });
}
