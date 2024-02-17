/* eslint-disable camelcase */
import { Logger } from 'browser-bunyan';
import { ServerParams } from '../../config.js';
import { BaseImgParams } from '../../types/params.js';
import { OnnxState, STATE_VERSION } from '../full.js';
import { Img2ImgSlice } from '../img2img.js';
import { InpaintSlice } from '../inpaint.js';
import { Txt2ImgSlice } from '../txt2img.js';
import { UpscaleSlice } from '../upscale.js';
import { DEFAULT_PROFILES } from '../profile.js';

// #region V13
export const V13 = 13;

export const REMOVED_KEYS_V13 = [] as const;

export type RemovedKeysV13 = typeof REMOVED_KEYS_V13[number];

export type AddedKeysV13 = 'txt2imgExperimental' | 'img2imgExperimental' | 'inpaintExperimental';
// #endregion

// #region V11
export const V11 = 11;

export const REMOVED_KEYS_V11 = ['tile', 'overlap'] as const;

export type RemovedKeysV11 = typeof REMOVED_KEYS_V11[number];

export type AddedKeysV11 = 'unet_tile' | 'unet_overlap' | 'vae_tile' | 'vae_overlap';

export type OnnxStateV11 = Omit<OnnxState, AddedKeysV13>;
// #endregion

// #region V7
export const V7 = 7;

export type BaseImgParamsV7<T extends BaseImgParams> = Omit<T, AddedKeysV11> & {
  overlap: number;
  tile: number;
};

export type OnnxStateV7 = Omit<OnnxState, 'img2img' | 'txt2img' | 'inpaint' | 'upscale'> & {
  img2img: BaseImgParamsV7<Img2ImgSlice['img2img']>;
  inpaint: BaseImgParamsV7<InpaintSlice['inpaint']>;
  txt2img: BaseImgParamsV7<Txt2ImgSlice['txt2img']>;
  upscale: BaseImgParamsV7<UpscaleSlice['upscale']>;
};
// #endregion

// add versions to this list as they are replaced
export type PreviousState = OnnxStateV7 | OnnxStateV11;

// always the latest version
export type CurrentState = OnnxState;

// any version of state
export type UnknownState = PreviousState | CurrentState;

export function applyStateMigrations(params: ServerParams, previousState: UnknownState, version: number, logger: Logger): OnnxState {
  logger.info('applying state migrations from version %s to version %s', version, STATE_VERSION);

  let migrated = previousState;

  if (version <= V7) {
    migrated = migrateV7ToV11(params, migrated as OnnxStateV7);
  }

  if (version <= V11) {
    migrated = migrateV11ToV13(params, migrated as OnnxStateV11);
  }

  return migrated as CurrentState;
}

export function migrateV7ToV11(params: ServerParams, previousState: OnnxStateV7): CurrentState {
  // add any missing keys
  const result: CurrentState = {
    ...params,
    ...previousState,
    img2img: {
      ...previousState.img2img,
      unet_overlap: params.unet_overlap.default,
      unet_tile: params.unet_tile.default,
      vae_overlap: params.vae_overlap.default,
      vae_tile: params.vae_tile.default,
    },
    inpaint: {
      ...previousState.inpaint,
      unet_overlap: params.unet_overlap.default,
      unet_tile: params.unet_tile.default,
      vae_overlap: params.vae_overlap.default,
      vae_tile: params.vae_tile.default,
    },
    txt2img: {
      ...previousState.txt2img,
      unet_overlap: params.unet_overlap.default,
      unet_tile: params.unet_tile.default,
      vae_overlap: params.vae_overlap.default,
      vae_tile: params.vae_tile.default,
    },
    upscale: {
      ...previousState.upscale,
      unet_overlap: params.unet_overlap.default,
      unet_tile: params.unet_tile.default,
      vae_overlap: params.vae_overlap.default,
      vae_tile: params.vae_tile.default,
    },
  };

  // add any missing profiles
  const existingProfiles = new Set(result.profiles.map((it) => it.name));
  for (const newProfile of DEFAULT_PROFILES) {
    if (existingProfiles.has(newProfile.name) === false) {
      result.profiles.push(newProfile);
    }
  }

  // TODO: remove extra keys

  return result;
}

export function migrateV11ToV13(params: ServerParams, previousState: OnnxStateV11): CurrentState {
  // add any missing keys
  const result: CurrentState = {
    ...params,
    ...previousState,
    txt2imgExperimental: {
      latentSymmetry: {
        enabled: false,
        gradientStart: 0,
        gradientEnd: 0,
        lineOfSymmetry: 0,
      },
      promptEditing: {
        enabled: false,
        filter: '',
        addSuffix: '',
        removeTokens: '',
      },
    },
    img2imgExperimental: {
      latentSymmetry: {
        enabled: false,
        gradientStart: 0,
        gradientEnd: 0,
        lineOfSymmetry: 0,
      },
      promptEditing: {
        enabled: false,
        filter: '',
        addSuffix: '',
        removeTokens: '',
      },
    },
    inpaintExperimental: {
      latentSymmetry: {
        enabled: false,
        gradientStart: 0,
        gradientEnd: 0,
        lineOfSymmetry: 0,
      },
      promptEditing: {
        enabled: false,
        filter: '',
        addSuffix: '',
        removeTokens: '',
      },
    },
  };

  return result;
}
