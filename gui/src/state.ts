/* eslint-disable max-lines */
/* eslint-disable no-null/no-null */
import { Maybe } from '@apextoaster/js-utils';
import { PaletteMode } from '@mui/material';
import { Logger } from 'noicejs';
import { createContext } from 'react';
import { StateCreator, StoreApi } from 'zustand';

import {
  ApiClient,
} from './client/base.js';
import { PipelineGrid } from './client/utils.js';
import { Config, ConfigFiles, ConfigState, ServerParams } from './config.js';
import { CorrectionModel, DiffusionModel, ExtraNetwork, ExtraSource, ExtrasFile, UpscalingModel } from './types/model.js';
import { ImageResponse, ReadyResponse, RetryParams } from './types/api.js';
import {
  BaseImgParams,
  BlendParams,
  BrushParams,
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintPixels,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from './types/params.js';

export const MISSING_INDEX = -1;

export type Theme = PaletteMode | ''; // tri-state, '' is unset

/**
 * Combine optional files and required ranges.
 */
export type TabState<TabParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;

export interface HistoryItem {
  image: ImageResponse;
  ready: Maybe<ReadyResponse>;
  retry: Maybe<RetryParams>;
}

export interface ProfileItem {
  name: string;
  params: BaseImgParams | Txt2ImgParams;
  highres?: Maybe<HighresParams>;
  upscale?: Maybe<UpscaleParams>;
}

interface DefaultSlice {
  defaults: TabState<BaseImgParams>;
  theme: Theme;

  setDefaults(param: Partial<BaseImgParams>): void;
  setTheme(theme: Theme): void;
}

interface HistorySlice {
  history: Array<HistoryItem>;
  limit: number;

  pushHistory(image: ImageResponse, retry?: RetryParams): void;
  removeHistory(image: ImageResponse): void;
  setLimit(limit: number): void;
  setReady(image: ImageResponse, ready: ReadyResponse): void;
}

interface ModelSlice {
  extras: ExtrasFile;

  removeCorrectionModel(model: CorrectionModel): void;
  removeDiffusionModel(model: DiffusionModel): void;
  removeExtraNetwork(model: ExtraNetwork): void;
  removeExtraSource(model: ExtraSource): void;
  removeUpscalingModel(model: UpscalingModel): void;

  setExtras(extras: Partial<ExtrasFile>): void;

  setCorrectionModel(model: CorrectionModel): void;
  setDiffusionModel(model: DiffusionModel): void;
  setExtraNetwork(model: ExtraNetwork): void;
  setExtraSource(model: ExtraSource): void;
  setUpscalingModel(model: UpscalingModel): void;
}

// #region tab slices
interface Txt2ImgSlice {
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

interface Img2ImgSlice {
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

interface InpaintSlice {
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

interface UpscaleSlice {
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

interface BlendSlice {
  blend: TabState<BlendParams>;
  blendBrush: BrushParams;
  blendModel: ModelParams;
  blendUpscale: UpscaleParams;

  resetBlend(): void;

  setBlend(blend: Partial<BlendParams>): void;
  setBlendBrush(brush: Partial<BrushParams>): void;
  setBlendModel(model: Partial<ModelParams>): void;
  setBlendUpscale(params: Partial<UpscaleParams>): void;
}

interface ResetSlice {
  resetAll(): void;
}

interface ProfileSlice {
  profiles: Array<ProfileItem>;

  removeProfile(profileName: string): void;

  saveProfile(profile: ProfileItem): void;
}
// #endregion

/**
 * Full merged state including all slices.
 */
export type OnnxState
  = DefaultSlice
  & HistorySlice
  & Img2ImgSlice
  & InpaintSlice
  & ModelSlice
  & Txt2ImgSlice
  & UpscaleSlice
  & BlendSlice
  & ResetSlice
  & ModelSlice
  & ProfileSlice;

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
 * React context binding for bunyan logger.
 */
export const LoggerContext = createContext<Maybe<Logger>>(undefined);

/**
 * React context binding for zustand state store.
 */
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);

/**
 * Key for zustand persistence, typically local storage.
 */
export const STATE_KEY = 'onnx-web';

/**
 * Current state version for zustand persistence.
 */
export const STATE_VERSION = 7;

export const BLEND_SOURCES = 2;

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
    batch: defaults.batch.default,
    cfg: defaults.cfg.default,
    eta: defaults.eta.default,
    negativePrompt: defaults.negativePrompt.default,
    prompt: defaults.prompt.default,
    scheduler: defaults.scheduler.default,
    steps: defaults.steps.default,
    seed: defaults.seed.default,
    tiledVAE: defaults.tiledVAE.default,
    tiles: defaults.tiles.default,
    overlap: defaults.overlap.default,
    stride: defaults.stride.default,
  };
}

/**
 * Prepare the state slice constructors.
 *
 * In the default state, image sources should be null and booleans should be false. Everything
 * else should be initialized from the default value in the base parameters.
 */
export function createStateSlices(server: ServerParams) {
  const defaultParams = baseParamsFromServer(server);
  const defaultHighres: HighresParams = {
    enabled: false,
    highresIterations: server.highresIterations.default,
    highresMethod: '',
    highresSteps: server.highresSteps.default,
    highresScale: server.highresScale.default,
    highresStrength: server.highresStrength.default,
  };
  const defaultModel: ModelParams = {
    control: server.control.default,
    correction: server.correction.default,
    model: server.model.default,
    pipeline: server.pipeline.default,
    platform: server.platform.default,
    upscaling: server.upscaling.default,
  };
  const defaultUpscale: UpscaleParams = {
    denoise: server.denoise.default,
    enabled: false,
    faces: false,
    faceOutscale: server.faceOutscale.default,
    faceStrength: server.faceStrength.default,
    outscale: server.outscale.default,
    scale: server.scale.default,
    upscaleOrder: server.upscaleOrder.default,
  };
  const defaultGrid: PipelineGrid = {
    enabled: false,
    columns: {
      parameter: 'seed',
      value: '',
    },
    rows: {
      parameter: 'seed',
      value: '',
    },
  };

  const createTxt2ImgSlice: Slice<Txt2ImgSlice> = (set) => ({
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
      }));
    },
    setTxt2ImgHighres(params) {
      set((prev) => ({
        txt2imgHighres: {
          ...prev.txt2imgHighres,
          ...params,
        },
      }));
    },
    setTxt2ImgModel(params) {
      set((prev) => ({
        txt2imgModel: {
          ...prev.txt2imgModel,
          ...params,
        },
      }));
    },
    setTxt2ImgUpscale(params) {
      set((prev) => ({
        txt2imgUpscale: {
          ...prev.txt2imgUpscale,
          ...params,
        },
      }));
    },
    setTxt2ImgVariable(params) {
      set((prev) => ({
        txt2imgVariable: {
          ...prev.txt2imgVariable,
          ...params,
        },
      }));
    },
    resetTxt2Img() {
      set({
        txt2img: {
          ...defaultParams,
          width: server.width.default,
          height: server.height.default,
        },
      });
    },
  });

  const createImg2ImgSlice: Slice<Img2ImgSlice> = (set) => ({
    img2img: {
      ...defaultParams,
      loopback: server.loopback.default,
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
    resetImg2Img() {
      set({
        img2img: {
          ...defaultParams,
          loopback: server.loopback.default,
          source: null,
          sourceFilter: '',
          strength: server.strength.default,
        },
      });
    },
    setImg2Img(params) {
      set((prev) => ({
        img2img: {
          ...prev.img2img,
          ...params,
        },
      }));
    },
    setImg2ImgHighres(params) {
      set((prev) => ({
        img2imgHighres: {
          ...prev.img2imgHighres,
          ...params,
        },
      }));
    },
    setImg2ImgModel(params) {
      set((prev) => ({
        img2imgModel: {
          ...prev.img2imgModel,
          ...params,
        },
      }));
    },
    setImg2ImgUpscale(params) {
      set((prev) => ({
        img2imgUpscale: {
          ...prev.img2imgUpscale,
          ...params,
        },
      }));
    },
  });

  const createInpaintSlice: Slice<InpaintSlice> = (set) => ({
    inpaint: {
      ...defaultParams,
      fillColor: server.fillColor.default,
      filter: server.filter.default,
      mask: null,
      noise: server.noise.default,
      source: null,
      strength: server.strength.default,
      tileOrder: server.tileOrder.default,
    },
    inpaintBrush: {
      ...DEFAULT_BRUSH,
    },
    inpaintHighres: {
      ...defaultHighres,
    },
    inpaintModel: {
      ...defaultModel,
    },
    inpaintUpscale: {
      ...defaultUpscale,
    },
    outpaint: {
      enabled: false,
      left: server.left.default,
      right: server.right.default,
      top: server.top.default,
      bottom: server.bottom.default,
    },
    resetInpaint() {
      set({
        inpaint: {
          ...defaultParams,
          fillColor: server.fillColor.default,
          filter: server.filter.default,
          mask: null,
          noise: server.noise.default,
          source: null,
          strength: server.strength.default,
          tileOrder: server.tileOrder.default,
        },
      });
    },
    setInpaint(params) {
      set((prev) => ({
        inpaint: {
          ...prev.inpaint,
          ...params,
        },
      }));
    },
    setInpaintBrush(brush) {
      set((prev) => ({
        inpaintBrush: {
          ...prev.inpaintBrush,
          ...brush,
        },
      }));
    },
    setInpaintHighres(params) {
      set((prev) => ({
        inpaintHighres: {
          ...prev.inpaintHighres,
          ...params,
        },
      }));
    },
    setInpaintModel(params) {
      set((prev) => ({
        inpaintModel: {
          ...prev.inpaintModel,
          ...params,
        },
      }));
    },
    setInpaintUpscale(params) {
      set((prev) => ({
        inpaintUpscale: {
          ...prev.inpaintUpscale,
          ...params,
        },
      }));
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

  const createHistorySlice: Slice<HistorySlice> = (set) => ({
    history: [],
    limit: DEFAULT_HISTORY.limit,
    pushHistory(image, retry) {
      set((prev) => ({
        ...prev,
        history: [
          {
            image,
            ready: undefined,
            retry,
          },
          ...prev.history,
        ].slice(0, prev.limit + DEFAULT_HISTORY.scrollback),
      }));
    },
    removeHistory(image) {
      set((prev) => ({
        ...prev,
        history: prev.history.filter((it) => it.image.outputs[0].key !== image.outputs[0].key),
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
        const history = [...prev.history];
        const idx = history.findIndex((it) => it.image.outputs[0].key === image.outputs[0].key);
        if (idx >= 0) {
          history[idx].ready = ready;
        } else {
          // TODO: error
        }

        return {
          ...prev,
          history,
        };
      });
    },
  });

  const createUpscaleSlice: Slice<UpscaleSlice> = (set) => ({
    upscale: {
      ...defaultParams,
      source: null,
    },
    upscaleHighres: {
      ...defaultHighres,
    },
    upscaleModel: {
      ...defaultModel,
    },
    upscaleUpscale: {
      ...defaultUpscale,
    },
    resetUpscale() {
      set({
        upscale: {
          ...defaultParams,
          source: null,
        },
      });
    },
    setUpscale(source) {
      set((prev) => ({
        upscale: {
          ...prev.upscale,
          ...source,
        },
      }));
    },
    setUpscaleHighres(params) {
      set((prev) => ({
        upscaleHighres: {
          ...prev.upscaleHighres,
          ...params,
        },
      }));
    },
    setUpscaleModel(params) {
      set((prev) => ({
        upscaleModel: {
          ...prev.upscaleModel,
          ...defaultModel,
        },
      }));
    },
    setUpscaleUpscale(params) {
      set((prev) => ({
        upscaleUpscale: {
          ...prev.upscaleUpscale,
          ...params,
        },
      }));
    },
  });

  const createBlendSlice: Slice<BlendSlice> = (set) => ({
    blend: {
      mask: null,
      sources: [],
    },
    blendBrush: {
      ...DEFAULT_BRUSH,
    },
    blendModel: {
      ...defaultModel,
    },
    blendUpscale: {
      ...defaultUpscale,
    },
    resetBlend() {
      set({
        blend: {
          mask: null,
          sources: [],
        },
      });
    },
    setBlend(blend) {
      set((prev) => ({
        blend: {
          ...prev.blend,
          ...blend,
        },
      }));
    },
    setBlendBrush(brush) {
      set((prev) => ({
        blendBrush: {
          ...prev.blendBrush,
          ...brush,
        },
      }));
    },
    setBlendModel(model) {
      set((prev) => ({
        blendModel: {
          ...prev.blendModel,
          ...model,
        },
      }));
    },
    setBlendUpscale(params) {
      set((prev) => ({
        blendUpscale: {
          ...prev.blendUpscale,
          ...params,
        },
      }));
    },
  });

  const createDefaultSlice: Slice<DefaultSlice> = (set) => ({
    defaults: {
      ...defaultParams,
    },
    theme: '',
    setDefaults(params) {
      set((prev) => ({
        defaults: {
          ...prev.defaults,
          ...params,
        }
      }));
    },
    setTheme(theme) {
      set((prev) => ({
        theme,
      }));
    }
  });

  const createResetSlice: Slice<ResetSlice> = (set) => ({
    resetAll() {
      set((prev) => {
        const next = { ...prev };
        next.resetImg2Img();
        next.resetInpaint();
        next.resetTxt2Img();
        next.resetUpscale();
        next.resetBlend();
        return next;
      });
    },
  });

  const createProfileSlice: Slice<ProfileSlice> = (set) => ({
    profiles: [],
    saveProfile(profile: ProfileItem) {
      set((prev) => {
        const profiles = [...prev.profiles];
        const idx = profiles.findIndex((it) => it.name === profile.name);
        if (idx >= 0) {
          profiles[idx] = profile;
        } else {
          profiles.push(profile);
        }
        return {
          ...prev,
          profiles,
        };
      });
    },
    removeProfile(profileName: string) {
      set((prev) => {
        const profiles = [...prev.profiles];
        const idx = profiles.findIndex((it) => it.name === profileName);
        if (idx >= 0) {
          profiles.splice(idx, 1);
        }
        return {
          ...prev,
          profiles,
        };
      });
    }
  });

  // eslint-disable-next-line sonarjs/cognitive-complexity
  const createModelSlice: Slice<ModelSlice> = (set) => ({
    extras: {
      correction: [],
      diffusion: [],
      networks: [],
      sources: [],
      upscaling: [],
    },
    setExtras(extras) {
      set((prev) => ({
        ...prev,
        extras: {
          ...prev.extras,
          ...extras,
        },
      }));
    },
    setCorrectionModel(model) {
      set((prev) => {
        const correction = [...prev.extras.correction];
        const exists = correction.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          correction.push(model);
        } else {
          correction[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            correction,
          },
        };
      });
    },
    setDiffusionModel(model) {
      set((prev) => {
        const diffusion = [...prev.extras.diffusion];
        const exists = diffusion.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          diffusion.push(model);
        } else {
          diffusion[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            diffusion,
          },
        };
      });
    },
    setExtraNetwork(model) {
      set((prev) => {
        const networks = [...prev.extras.networks];
        const exists = networks.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          networks.push(model);
        } else {
          networks[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            networks,
          },
        };
      });
    },
    setExtraSource(model) {
      set((prev) => {
        const sources = [...prev.extras.sources];
        const exists = sources.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          sources.push(model);
        } else {
          sources[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            sources,
          },
        };
      });
    },
    setUpscalingModel(model) {
      set((prev) => {
        const upscaling = [...prev.extras.upscaling];
        const exists = upscaling.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          upscaling.push(model);
        } else {
          upscaling[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            upscaling,
          },
        };
      });
    },
    removeCorrectionModel(model) {
      set((prev) => {
        const correction = prev.extras.correction.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            correction,
          },
        };
      });

    },
    removeDiffusionModel(model) {
      set((prev) => {
        const diffusion = prev.extras.diffusion.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            diffusion,
          },
        };
      });

    },
    removeExtraNetwork(model) {
      set((prev) => {
        const networks = prev.extras.networks.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            networks,
          },
        };
      });

    },
    removeExtraSource(model) {
      set((prev) => {
        const sources = prev.extras.sources.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            sources,
          },
        };
      });

    },
    removeUpscalingModel(model) {
      set((prev) => {
        const upscaling = prev.extras.upscaling.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            upscaling,
          },
        };
      });
    },
  });

  return {
    createDefaultSlice,
    createHistorySlice,
    createImg2ImgSlice,
    createInpaintSlice,
    createTxt2ImgSlice,
    createUpscaleSlice,
    createBlendSlice,
    createResetSlice,
    createModelSlice,
    createProfileSlice,
  };
}
