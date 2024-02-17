/* eslint-disable camelcase */
import { Maybe } from '@apextoaster/js-utils';
import { Logger } from 'noicejs';
import { createContext } from 'react';
import { StoreApi } from 'zustand';

import {
  ApiClient,
} from '../client/base.js';
import { PipelineGrid } from '../client/utils.js';
import { Config, ServerParams } from '../config.js';
import { BlendSlice, createBlendSlice } from './blend.js';
import { DefaultSlice, createDefaultSlice } from './default.js';
import { HistorySlice, createHistorySlice } from './history.js';
import { Img2ImgSlice, createImg2ImgSlice } from './img2img.js';
import { InpaintSlice, createInpaintSlice } from './inpaint.js';
import { ModelSlice, createModelSlice } from './model.js';
import { ProfileSlice, createProfileSlice } from './profile.js';
import { ResetSlice, createResetSlice } from './reset.js';
import { SettingsSlice, createSettingsSlice } from './settings.js';
import { Txt2ImgSlice, createTxt2ImgSlice } from './txt2img.js';
import { UpscaleSlice, createUpscaleSlice } from './upscale.js';
import {
  BaseImgParams,
  ExperimentalParams,
  HighresParams,
  ModelParams,
  UpscaleParams,
} from '../types/params.js';

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
  & ProfileSlice
  & SettingsSlice;

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
export const STATE_VERSION = 13;

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
    tiled_vae: defaults.tiled_vae.default,
    unet_overlap: defaults.unet_overlap.default,
    unet_tile: defaults.unet_tile.default,
    vae_overlap: defaults.vae_overlap.default,
    vae_tile: defaults.vae_tile.default,
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
  const defaultExperimental: ExperimentalParams = {
    promptEditing: {
      enabled: false,
      filter: '',
      addSuffix  : '',
      removeTokens: '',
    },
    latentSymmetry: {
      enabled: false,
      gradientStart: 0,
      gradientEnd: 0,
      lineOfSymmetry: 0,
    },
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

  return {
    createBlendSlice: createBlendSlice(defaultModel, defaultUpscale),
    createDefaultSlice: createDefaultSlice(defaultParams),
    createHistorySlice: createHistorySlice(),
    createImg2ImgSlice: createImg2ImgSlice(server, defaultParams, defaultHighres, defaultModel, defaultUpscale, defaultExperimental),
    createInpaintSlice: createInpaintSlice(server, defaultParams, defaultHighres, defaultModel, defaultUpscale, defaultExperimental),
    createModelSlice: createModelSlice(),
    createProfileSlice: createProfileSlice(),
    createResetSlice: createResetSlice(),
    createSettingsSlice: createSettingsSlice(),
    createTxt2ImgSlice: createTxt2ImgSlice(server, defaultParams, defaultHighres, defaultModel, defaultUpscale, defaultExperimental, defaultGrid),
    createUpscaleSlice: createUpscaleSlice(defaultParams, defaultHighres, defaultModel, defaultUpscale),
  };
}
