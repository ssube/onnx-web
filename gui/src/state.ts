/* eslint-disable no-null/no-null */
import { doesExist, Maybe } from '@apextoaster/js-utils';
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
  ReadyResponse,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from './client.js';
import { Config, ConfigFiles, ConfigState, ServerParams } from './config.js';

/**
 * Combine optional files and required ranges.
 */
type TabState<TabParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;

interface LoadingItem {
  image: ImageResponse;
  ready: Maybe<ReadyResponse>;
}

interface BrushSlice {
  brush: BrushParams;

  setBrush(brush: Partial<BrushParams>): void;
}

interface DefaultSlice {
  defaults: TabState<BaseImgParams>;

  setDefaults(param: Partial<BaseImgParams>): void;
}

interface HistorySlice {
  history: Array<ImageResponse>;
  limit: number;
  loading: Array<LoadingItem>;

  clearLoading(image: ImageResponse): void;
  pushHistory(image: ImageResponse): void;
  pushLoading(image: ImageResponse): void;
  removeHistory(image: ImageResponse): void;
  setLimit(limit: number): void;
  /**
   * @todo should check ready and move the image from loading to history
   */
  setReady(image: ImageResponse, ready: ReadyResponse): void;
}

interface ModelSlice {
  model: ModelParams;

  setModel(model: Partial<ModelParams>): void;
}

// #region tab slices
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

interface OutpaintSlice {
  outpaint: OutpaintPixels;

  setOutpaint(pixels: Partial<OutpaintPixels>): void;
}

interface UpscaleSlice {
  upscale: UpscaleParams;
  upscaleTab: TabState<UpscaleReqParams>;

  setUpscale(upscale: Partial<UpscaleParams>): void;
  setUpscaleTab(params: Partial<UpscaleReqParams>): void;
  resetUpscaleTab(): void;
}
// #endregion

/**
 * Full merged state including all slices.
 */
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

/**
 * Shorthand for state creator to reduce repeated arguments.
 */
export type Slice<T> = StateCreator<OnnxState, [], [], T>;

/**
 * React context binding for API client.
 */
export const ClientContext = createContext<Maybe<ApiClient>>(undefined);

/**
 * React context binding for merged config, including server parameters.
 */
export const ConfigContext = createContext<Maybe<Config<ServerParams>>>(undefined);

/**
 * React context binding for zustand state store.
 */
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);

/**
 * Current state version for zustand persistence.
 */
export const STATE_VERSION = 5;

/**
 * Default parameters for the inpaint brush.
 *
 * Not provided by the server yet.
 */
export const DEFAULT_BRUSH = {
  color: 255,
  size: 8,
  strength: 0.5,
};

/**
 * Default parameters for the image history.
 *
 * Not provided by the server yet.
 */
export const DEFAULT_HISTORY = {
  /**
   * The number of images to be shown.
   */
  limit: 4,

  /**
   * The number of additional images to be kept in history, so they can scroll
   * back into view when you delete one. Does not include deleted images.
   */
  scrollback: 2,
};

export function baseParamsFromServer(defaults: ServerParams): Required<BaseImgParams> {
  return {
    cfg: defaults.cfg.default,
    negativePrompt: defaults.negativePrompt.default,
    prompt: defaults.prompt.default,
    scheduler: defaults.scheduler.default,
    steps: defaults.steps.default,
    seed: defaults.seed.default,
  };
}

/**
 * Prepare the state slice constructors.
 *
 * In the default state, image sources should be null and booleans should be false. Everything
 * else should be initialized from the default value in the base parameters.
 */
export function createStateSlices(server: ServerParams) {
  const base = baseParamsFromServer(server);

  const createTxt2ImgSlice: Slice<Txt2ImgSlice> = (set) => ({
    txt2img: {
      ...base,
      width: server.width.default,
      height: server.height.default,
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
          ...base,
          width: server.width.default,
          height: server.height.default,
        },
      });
    },
  });

  const createImg2ImgSlice: Slice<Img2ImgSlice> = (set) => ({
    img2img: {
      ...base,
      source: null,
      strength: server.strength.default,
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
          ...base,
          source: null,
          strength: server.strength.default,
        },
      });
    },
  });

  const createInpaintSlice: Slice<InpaintSlice> = (set) => ({
    inpaint: {
      ...base,
      fillColor: server.fillColor.default,
      filter: server.filter.default,
      mask: null,
      noise: server.noise.default,
      source: null,
      strength: server.strength.default,
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
          ...base,
          fillColor: server.fillColor.default,
          filter: server.filter.default,
          mask: null,
          noise: server.noise.default,
          source: null,
          strength: server.strength.default,
        },
      });
    },
  });

  const createHistorySlice: Slice<HistorySlice> = (set) => ({
    history: [],
    limit: DEFAULT_HISTORY.limit,
    loading: [],
    clearLoading(image) {
      set((prev) => ({
        ...prev,
        loading: prev.loading.filter((it) => it.image.output.key !== image.output.key),
      }));
    },
    pushHistory(image) {
      set((prev) => ({
        ...prev,
        history: [
          image,
          ...prev.history,
        ].slice(0, prev.limit + DEFAULT_HISTORY.scrollback),
        loading: prev.loading.filter((it) => it.image.output.key !== image.output.key),
      }));
    },
    pushLoading(image) {
      set((prev) => ({
        ...prev,
        loading: [
          {
            image,
            ready: {
              progress: 0,
              ready: false,
            },
          },
          ...prev.loading,
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
    setReady(image, ready) {
      set((prev) => {
        const loading = [...prev.loading];
        const idx = loading.findIndex((it) => it.image.output.key === image.output.key);
        if (idx >= 0) {
          loading[idx].ready = ready;
        } else {
          // TODO: error
        }

        return {
          ...prev,
          loading,
        };
      });
    },
  });

  const createOutpaintSlice: Slice<OutpaintSlice> = (set) => ({
    outpaint: {
      enabled: false,
      left: server.left.default,
      right: server.right.default,
      top: server.top.default,
      bottom: server.bottom.default,
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

  const createBrushSlice: Slice<BrushSlice> = (set) => ({
    brush: {
      ...DEFAULT_BRUSH,
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

  const createUpscaleSlice: Slice<UpscaleSlice> = (set) => ({
    upscale: {
      denoise: server.denoise.default,
      enabled: false,
      faces: false,
      scale: server.scale.default,
      outscale: server.outscale.default,
      faceStrength: server.faceStrength.default,
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

  const createDefaultSlice: Slice<DefaultSlice> = (set) => ({
    defaults: {
      ...base,
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

  const createModelSlice: Slice<ModelSlice> = (set) => ({
    model: {
      model: server.model.default,
      platform: server.platform.default,
      upscaling: server.upscaling.default,
      correction: server.correction.default,
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
