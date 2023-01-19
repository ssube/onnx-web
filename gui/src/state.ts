/* eslint-disable no-null/no-null */
import { Maybe } from '@apextoaster/js-utils';
import { createContext } from 'react';
import { StateCreator, StoreApi } from 'zustand';

import {
  ApiClient,
  BaseImgParams,
  BrushParams,
  ImageResponse,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintPixels,
  paramsFromConfig,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from './client.js';
import { Config, ConfigFiles, ConfigState, ServerParams } from './config.js';

type TabState<TabParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;

interface Txt2ImgSlice {
  txt2img: TabState<Txt2ImgParams>;

  setTxt2Img(params: Partial<Txt2ImgParams>): void;
  resetTxt2Img(): void;
}

interface Img2ImgSlice {
  img2img: TabState<Img2ImgParams>;

  setImg2Img(params: Partial<Img2ImgParams>): void;
  resetImg2Img(): void;
}

interface InpaintSlice {
  inpaint: TabState<InpaintParams>;

  setInpaint(params: Partial<InpaintParams>): void;
  resetInpaint(): void;
}

interface HistorySlice {
  history: Array<ImageResponse>;
  limit: number;
  loading: Maybe<ImageResponse>;

  pushHistory(image: ImageResponse): void;
  removeHistory(image: ImageResponse): void;
  setLimit(limit: number): void;
  setLoading(image: Maybe<ImageResponse>): void;
}

interface DefaultSlice {
  defaults: TabState<BaseImgParams>;

  setDefaults(param: Partial<BaseImgParams>): void;
}

interface OutpaintSlice {
  outpaint: OutpaintPixels;

  setOutpaint(pixels: Partial<OutpaintPixels>): void;
}

interface BrushSlice {
  brush: BrushParams;

  setBrush(brush: Partial<BrushParams>): void;
}

interface UpscaleSlice {
  upscale: UpscaleParams;
  upscaleTab: TabState<UpscaleReqParams>;

  setUpscale(upscale: Partial<UpscaleParams>): void;
  setUpscaleTab(params: Partial<UpscaleReqParams>): void;
  resetUpscaleTab(): void;
}

interface ModelSlice {
  model: ModelParams;

  setModel(model: Partial<ModelParams>): void;
}

export type OnnxState
  = BrushSlice
  & DefaultSlice
  & HistorySlice
  & Img2ImgSlice
  & InpaintSlice
  & ModelSlice
  & OutpaintSlice
  & Txt2ImgSlice
  & UpscaleSlice;

export function createStateSlices(base: ServerParams) {
  const defaults = paramsFromConfig(base);

  const createTxt2ImgSlice: StateCreator<OnnxState, [], [], Txt2ImgSlice> = (set) => ({
    txt2img: {
      ...defaults,
      width: base.width.default,
      height: base.height.default,
    },
    setTxt2Img(params) {
      set((prev) => ({
        txt2img: {
          ...prev.txt2img,
          ...params,
        },
      }));
    },
    resetTxt2Img() {
      set({
        txt2img: {
          ...defaults,
          width: base.width.default,
          height: base.height.default,
        },
      });
    },
  });

  const createImg2ImgSlice: StateCreator<OnnxState, [], [], Img2ImgSlice> = (set) => ({
    img2img: {
      ...defaults,
      source: null,
      strength: base.strength.default,
    },
    setImg2Img(params) {
      set((prev) => ({
        img2img: {
          ...prev.img2img,
          ...params,
        },
      }));
    },
    resetImg2Img() {
      set({
        img2img: {
          ...defaults,
          source: null,
          strength: base.strength.default,
        },
      });
    },
  });

  const createInpaintSlice: StateCreator<OnnxState, [], [], InpaintSlice> = (set) => ({
    inpaint: {
      ...defaults,
      fillColor: '',
      filter: 'none',
      mask: null,
      noise: 'histogram',
      source: null,
      strength: 1.0,
    },
    setInpaint(params) {
      set((prev) => ({
        inpaint: {
          ...prev.inpaint,
          ...params,
        },
      }));
    },
    resetInpaint() {
      set({
        inpaint: {
          ...defaults,
          fillColor: '',
          filter: 'none',
          mask: null,
          noise: 'histogram',
          source: null,
          strength: 1.0,
        },
      });
    },
  });

  const createHistorySlice: StateCreator<OnnxState, [], [], HistorySlice> = (set) => ({
    history: [],
    limit: 4,
    loading: null,
    pushHistory(image) {
      set((prev) => ({
        ...prev,
        history: [
          image,
          ...prev.history,
        ].slice(0, prev.limit),
        loading: null,
      }));
    },
    removeHistory(image) {
      set((prev) => ({
        ...prev,
        history: prev.history.filter((it) => it.output !== image.output),
      }));
    },
    setLimit(limit) {
      set((prev) => ({
        ...prev,
        limit,
      }));
    },
    setLoading(loading) {
      set((prev) => ({
        ...prev,
        loading,
      }));
    },
  });

  const createOutpaintSlice: StateCreator<OnnxState, [], [], OutpaintSlice> = (set) => ({
    outpaint: {
      enabled: false,
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
    },
    setOutpaint(pixels) {
      set((prev) => ({
        outpaint: {
          ...prev.outpaint,
          ...pixels,
        }
      }));
    },
  });

  const createBrushSlice: StateCreator<OnnxState, [], [], BrushSlice> = (set) => ({
    brush: {
      color: 255,
      size: 8,
      strength: 0.5,
    },
    setBrush(brush) {
      set((prev) => ({
        brush: {
          ...prev.brush,
          ...brush,
        },
      }));
    },
  });

  const createUpscaleSlice: StateCreator<OnnxState, [], [], UpscaleSlice> = (set) => ({
    upscale: {
      denoise: 0.5,
      enabled: false,
      faces: false,
      scale: 1,
      outscale: 1,
      faceStrength: 0.5,
    },
    upscaleTab: {
      source: null,
    },
    setUpscale(upscale) {
      set((prev) => ({
        upscale: {
          ...prev.upscale,
          ...upscale,
        },
      }));
    },
    setUpscaleTab(source) {
      set((prev) => ({
        upscaleTab: {
          ...prev.upscaleTab,
          ...source,
        },
      }));
    },
    resetUpscaleTab() {
      set({
        upscaleTab: {
          source: null,
        },
      });
    },
  });

  const createDefaultSlice: StateCreator<OnnxState, [], [], DefaultSlice> = (set) => ({
    defaults: {
      ...defaults,
    },
    setDefaults(params) {
      set((prev) => ({
        defaults: {
          ...prev.defaults,
          ...params,
        }
      }));
    },
  });

  const createModelSlice: StateCreator<OnnxState, [], [], ModelSlice> = (set) => ({
    model: {
      model: '',
      platform: '',
      upscaling: '',
      correction: '',
    },
    setModel(params) {
      set((prev) => ({
        model: {
          ...prev.model,
          ...params,
        }
      }));
    },
  });

  return {
    createBrushSlice,
    createDefaultSlice,
    createHistorySlice,
    createImg2ImgSlice,
    createInpaintSlice,
    createModelSlice,
    createOutpaintSlice,
    createTxt2ImgSlice,
    createUpscaleSlice,
  };
}

export const ClientContext = createContext<Maybe<ApiClient>>(undefined);
export const ConfigContext = createContext<Maybe<Config<ServerParams>>>(undefined);
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);
