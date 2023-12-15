import { BlendSlice } from './blend.js';
import { Img2ImgSlice } from './img2img.js';
import { InpaintSlice } from './inpaint.js';
import { Txt2ImgSlice } from './txt2img.js';
import { Slice } from './types.js';
import { UpscaleSlice } from './upscale.js';

export type SlicesWithReset = Txt2ImgSlice & Img2ImgSlice & InpaintSlice & UpscaleSlice & BlendSlice;

export interface ResetSlice {
  resetAll(): void;
}

export function createResetSlice<TState extends ResetSlice & SlicesWithReset>(): Slice<TState, ResetSlice> {
  return (set) => ({
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
}
