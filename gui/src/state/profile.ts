/* eslint-disable camelcase */
import { Maybe } from '@apextoaster/js-utils';
import { BaseImgParams, HighresParams, Txt2ImgParams, UpscaleParams } from '../types/params.js';
import { Slice } from './types.js';

export interface ProfileItem {
  name: string;
  params: BaseImgParams | Txt2ImgParams;
  highres?: Maybe<HighresParams>;
  upscale?: Maybe<UpscaleParams>;
}

export interface ProfileSlice {
  profiles: Array<ProfileItem>;

  removeProfile(profileName: string): void;

  saveProfile(profile: ProfileItem): void;
}

export function createProfileSlice<TState extends ProfileSlice>(): Slice<TState, ProfileSlice> {
  return (set) => ({
    profiles: [...DEFAULT_PROFILES],
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
}

export const DEFAULT_HIGHRES_ON: HighresParams = {
  enabled: true,
  highresIterations: 1,
  highresMethod: 'upscale',
  highresSteps: 150,
  highresScale: 2,
  highresStrength: 0.2,
};

export const DEFAULT_HIGHRES_OFF: HighresParams = {
  ...DEFAULT_HIGHRES_ON,
  enabled: false,
};

export const DEFAULT_UPSCALE_OFF: UpscaleParams = {
  denoise: 0.5,
  enabled: false,
  faces: false,
  faceOutscale: 1,
  faceStrength: 0.5,
  outscale: 1,
  scale: 1,
  upscaleOrder: 'correction-first',
};

export const DEFAULT_PROFILES: Array<ProfileItem> = [
  // SD v1.5 base
  {
    name: 'base SD v1.5',
    params: {
      batch: 1,
      cfg: 5,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'deis',
      steps: 30,
      seed: -1,
      tiled_vae: false,
      unet_overlap: 0.75,
      unet_tile: 512,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 512,
      height: 512
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SD v1.5 LCM
  {
    name: 'base SD v1.5 LCM',
    params: {
      // "pipeline": "txt2img-sdxl",
      scheduler: 'lcm',
      prompt: '<lora:lcm:1.0> ',
      negativePrompt: '',
      cfg: 1.5,
      seed: -1,
      steps: 12,
      eta: 0.0,
      batch: 1,
      tiled_vae: false,
      unet_overlap: 0.5,
      unet_tile: 512,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 512,
      height: 512,
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },  // SD v1.5 highres
  {
    name: 'base SD v1.5 highres',
    params: {
      batch: 1,
      cfg: 5,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'deis',
      steps: 30,
      seed: -1,
      tiled_vae: false,
      unet_overlap: 0.75,
      unet_tile: 512,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 512,
      height: 512
    },
    highres: DEFAULT_HIGHRES_ON,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SD v1.5 panorama
  {
    name: 'base SD v1.5 panorama',
    params: {
      batch: 1,
      cfg: 12,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'ddim',
      steps: 125,
      seed: -1,
      tiled_vae: true,
      unet_overlap: 0.75,
      unet_tile: 512,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 1024,
      height: 512
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SDXL base
  {
    name: 'base SDXL',
    params: {
      batch: 1,
      cfg: 10,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'dpm-sde',
      steps: 120,
      seed: -1,
      tiled_vae: false,
      unet_overlap: 0.75,
      unet_tile: 1024,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 1024,
      height: 1024,
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SDXL highres
  {
    name: 'base SDXL highres',
    params: {
      batch: 1,
      cfg: 10,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'dpm-sde',
      steps: 120,
      seed: -1,
      tiled_vae: false,
      unet_overlap: 0.75,
      unet_tile: 1024,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 1024,
      height: 1024,
    },
    highres: DEFAULT_HIGHRES_ON,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SDXL LCM
  {
    name: 'base SDXL LCM',
    params: {
      // "pipeline": "txt2img-sdxl",
      scheduler: 'lcm',
      prompt: '<lora:sdxl-lcm:1.0> ',
      negativePrompt: '',
      cfg: 1.5,
      seed: -1,
      steps: 12,
      eta: 0.0,
      batch: 1,
      tiled_vae: false,
      unet_overlap: 0.5,
      unet_tile: 1024,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 1024,
      height: 1024,
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SDXL panorama
  {
    name: 'base SDXL panorama',
    params: {
      batch: 1,
      cfg: 12,
      eta: 0,
      negativePrompt: '',
      prompt: '',
      scheduler: 'ddim',
      steps: 125,
      seed: -1,
      tiled_vae: true,
      unet_overlap: 0.75,
      unet_tile: 1024,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 2048,
      height: 1024,
    },
    highres: DEFAULT_HIGHRES_ON,
    upscale: DEFAULT_UPSCALE_OFF,
  },
  // SDXL turbo
  {
    name: 'base SDXL turbo',
    params: {
      scheduler: 'dpm-sde',
      prompt: '',
      negativePrompt: '',
      cfg: 1.5,
      seed: -1,
      steps: 6,
      eta: 0.0,
      batch: 1,
      tiled_vae: false,
      unet_overlap: 0.75,
      unet_tile: 768,
      vae_overlap: 0.25,
      vae_tile: 512,
      width: 512,
      height: 768,
    },
    highres: DEFAULT_HIGHRES_OFF,
    upscale: DEFAULT_UPSCALE_OFF,
  },
];
