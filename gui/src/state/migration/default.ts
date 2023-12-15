/* eslint-disable camelcase */
import { ServerParams } from '../../config.js';
import { BaseImgParams } from '../../types/params.js';
import { OnnxState, STATE_VERSION } from '../full.js';
import { Img2ImgSlice } from '../img2img.js';
import { InpaintSlice } from '../inpaint.js';
import { Txt2ImgSlice } from '../txt2img.js';
import { UpscaleSlice } from '../upscale.js';

export const REMOVE_KEYS = ['tile', 'overlap'] as const;

export type RemovedKeys = typeof REMOVE_KEYS[number];

// TODO: can the compiler calculate this?
export type AddedKeysV11 = 'unet_tile' | 'unet_overlap' | 'vae_tile' | 'vae_overlap';

export type BaseImgParamsV7<T extends BaseImgParams> = Omit<T, AddedKeysV11> & {
  overlap: number;
  tile: number;
};

export type OnnxStateV7 = Omit<OnnxState, 'img2img' | 'txt2img'> & {
  img2img: BaseImgParamsV7<Img2ImgSlice['img2img']>;
  inpaint: BaseImgParamsV7<InpaintSlice['inpaint']>;
  txt2img: BaseImgParamsV7<Txt2ImgSlice['txt2img']>;
  upscale: BaseImgParamsV7<UpscaleSlice['upscale']>;
};

export type PreviousState = OnnxStateV7;
export type CurrentState = OnnxState;
export type UnknownState = PreviousState | CurrentState;

export function applyStateMigrations(params: ServerParams, previousState: UnknownState, version: number): OnnxState {
  // eslint-disable-next-line no-console
  console.log('applying migrations from %s to %s', version, STATE_VERSION);

  if (version < STATE_VERSION) {
    return migrateDefaults(params, previousState as PreviousState);
  }

  return previousState as CurrentState;
}

export function migrateDefaults(params: ServerParams, previousState: PreviousState): CurrentState {
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

  // TODO: remove extra keys

  return result;
}
