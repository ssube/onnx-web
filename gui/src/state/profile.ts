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
}
