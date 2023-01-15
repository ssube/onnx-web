/* eslint-disable no-null/no-null */
import { Maybe } from '@apextoaster/js-utils';
import { createContext } from 'react';
import { StateCreator, StoreApi } from 'zustand';

import {
  ApiClient,
  ApiResponse,
  BaseImgParams,
  BrushParams,
  Img2ImgParams,
  InpaintParams,
  OutpaintPixels,
  paramsFromConfig,
  Txt2ImgParams,
} from './api/client.js';
import { ConfigFiles, ConfigParams, ConfigState } from './config.js';

type TabState<TabParams extends BaseImgParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;

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
  history: Array<ApiResponse>;
  limit: number;
  loading: Maybe<ApiResponse>;

  pushHistory(image: ApiResponse): void;
  removeHistory(image: ApiResponse): void;
  setLimit(limit: number): void;
  setLoading(image: Maybe<ApiResponse>): void;
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

export type OnnxState = Txt2ImgSlice & Img2ImgSlice & InpaintSlice & HistorySlice & DefaultSlice & OutpaintSlice & BrushSlice;

export function createStateSlices(base: ConfigParams) {
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
      filter: 'none',
      mask: null,
      noise: 'histogram',
      source: null,
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
          filter: 'none',
          mask: null,
          noise: 'histogram',
          source: null,
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
        ],
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

  return {
    createDefaultSlice,
    createHistorySlice,
    createImg2ImgSlice,
    createInpaintSlice,
    createTxt2ImgSlice,
    createOutpaintSlice,
    createBrushSlice,
  };
}

export const ClientContext = createContext<Maybe<ApiClient>>(undefined);
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);
