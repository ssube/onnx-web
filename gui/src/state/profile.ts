import { Maybe } from '@apextoaster/js-utils';
import { BaseImgParams, HighresParams, Txt2ImgParams, UpscaleParams } from '../types/params.js';

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
