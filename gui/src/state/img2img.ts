
import { ServerParams } from '../config.js';
import {
  BaseImgParams,
  ExperimentalParams,
  HighresParams,
  Img2ImgParams,
  ModelParams,
  UpscaleParams,
} from '../types/params.js';
import { Slice, TabState } from './types.js';

export interface Img2ImgSlice {
  img2img: TabState<Img2ImgParams>;
  img2imgModel: ModelParams;
  img2imgHighres: HighresParams;
  img2imgUpscale: UpscaleParams;
  img2imgExperimental: ExperimentalParams;

  resetImg2Img(): void;

  setImg2Img(params: Partial<Img2ImgParams>): void;
  setImg2ImgModel(params: Partial<ModelParams>): void;
  setImg2ImgHighres(params: Partial<HighresParams>): void;
  setImg2ImgUpscale(params: Partial<UpscaleParams>): void;
  setImg2ImgExperimental(params: Partial<ExperimentalParams>): void;
}

// eslint-disable-next-line max-params
export function createImg2ImgSlice<TState extends Img2ImgSlice>(
  server: ServerParams,
  defaultParams: Required<BaseImgParams>,
  defaultHighres: HighresParams,
  defaultModel: ModelParams,
  defaultUpscale: UpscaleParams,
  defaultExperimental: ExperimentalParams,
): Slice<TState, Img2ImgSlice> {
  return (set) => ({
    img2img: {
      ...defaultParams,
      loopback: server.loopback.default,
      // eslint-disable-next-line no-null/no-null
      source: null,
      sourceFilter: '',
      strength: server.strength.default,
    },
    img2imgHighres: {
      ...defaultHighres,
    },
    img2imgModel: {
      ...defaultModel,
    },
    img2imgUpscale: {
      ...defaultUpscale,
    },
    img2imgExperimental: {
      ...defaultExperimental,
    },
    resetImg2Img() {
      set({
        img2img: {
          ...defaultParams,
          loopback: server.loopback.default,
          // eslint-disable-next-line no-null/no-null
          source: null,
          sourceFilter: '',
          strength: server.strength.default,
        },
      } as Partial<TState>);
    },
    setImg2Img(params) {
      set((prev) => ({
        img2img: {
          ...prev.img2img,
          ...params,
        },
      } as Partial<TState>));
    },
    setImg2ImgHighres(params) {
      set((prev) => ({
        img2imgHighres: {
          ...prev.img2imgHighres,
          ...params,
        },
      } as Partial<TState>));
    },
    setImg2ImgModel(params) {
      set((prev) => ({
        img2imgModel: {
          ...prev.img2imgModel,
          ...params,
        },
      } as Partial<TState>));
    },
    setImg2ImgUpscale(params) {
      set((prev) => ({
        img2imgUpscale: {
          ...prev.img2imgUpscale,
          ...params,
        },
      } as Partial<TState>));
    },
    setImg2ImgExperimental(params) {
      set((prev) => ({
        img2imgExperimental: {
          ...prev.img2imgExperimental,
          ...params,
        },
      } as Partial<TState>));
    },
  });
}
