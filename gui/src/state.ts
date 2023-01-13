import { Maybe } from '@apextoaster/js-utils';
import { createContext } from 'react';
import { StateCreator, StoreApi } from 'zustand';

import {
  ApiClient,
  ApiResponse,
  BaseImgParams,
  Img2ImgParams,
  InpaintParams,
  paramsFromConfig,
  Txt2ImgParams,
} from './api/client.js';
import { ConfigParams, ConfigState } from './config.js';

type TabState<TabParams extends BaseImgParams> = ConfigState<Required<TabParams>>;

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
  loading: boolean;

  pushHistory(image: ApiResponse): void;
  removeHistory(image: ApiResponse): void;
  setLimit(limit: number): void;
  setLoading(loading: boolean): void;
}

interface DefaultSlice {
  defaults: TabState<BaseImgParams>;

  setDefaults(param: Partial<BaseImgParams>): void;
}

export type OnnxState = Txt2ImgSlice & Img2ImgSlice & InpaintSlice & HistorySlice & DefaultSlice;

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
          strength: base.strength.default,
        },
      });
    },
  });

  const createInpaintSlice: StateCreator<OnnxState, [], [], InpaintSlice> = (set) => ({
    inpaint: {
      ...defaults,
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
        },
      });
    },
  });

  const createHistorySlice: StateCreator<OnnxState, [], [], HistorySlice> = (set) => ({
    history: [],
    limit: 4,
    loading: false,
    pushHistory(image) {
      set((prev) => ({
        ...prev,
        history: [
          image,
          ...prev.history,
        ],
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
  };
}

export const ClientContext = createContext<Maybe<ApiClient>>(undefined);
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);
