import { ServerParams } from '../config.js';
import {
  BaseImgParams,
  BrushParams,
  HighresParams,
  InpaintParams,
  ModelParams,
  OutpaintPixels,
  UpscaleParams,
} from '../types/params.js';
import { DEFAULT_BRUSH, Slice, TabState } from './types.js';
export interface InpaintSlice {
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

// eslint-disable-next-line max-params
export function createInpaintSlice<TState extends InpaintSlice>(
  server: ServerParams,
  defaultParams: Required<BaseImgParams>,
  defaultHighres: HighresParams,
  defaultModel: ModelParams,
  defaultUpscale: UpscaleParams,
): Slice<TState, InpaintSlice> {
  return (set) => ({
    inpaint: {
      ...defaultParams,
      fillColor: server.fillColor.default,
      filter: server.filter.default,
      // eslint-disable-next-line no-null/no-null
      mask: null,
      noise: server.noise.default,
      // eslint-disable-next-line no-null/no-null
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
          // eslint-disable-next-line no-null/no-null
          mask: null,
          noise: server.noise.default,
          // eslint-disable-next-line no-null/no-null
          source: null,
          strength: server.strength.default,
          tileOrder: server.tileOrder.default,
        },
      } as Partial<TState>);
    },
    setInpaint(params) {
      set((prev) => ({
        inpaint: {
          ...prev.inpaint,
          ...params,
        },
      } as Partial<TState>));
    },
    setInpaintBrush(brush) {
      set((prev) => ({
        inpaintBrush: {
          ...prev.inpaintBrush,
          ...brush,
        },
      } as Partial<TState>));
    },
    setInpaintHighres(params) {
      set((prev) => ({
        inpaintHighres: {
          ...prev.inpaintHighres,
          ...params,
        },
      } as Partial<TState>));
    },
    setInpaintModel(params) {
      set((prev) => ({
        inpaintModel: {
          ...prev.inpaintModel,
          ...params,
        },
      } as Partial<TState>));
    },
    setInpaintUpscale(params) {
      set((prev) => ({
        inpaintUpscale: {
          ...prev.inpaintUpscale,
          ...params,
        },
      } as Partial<TState>));
    },
    setOutpaint(pixels) {
      set((prev) => ({
        outpaint: {
          ...prev.outpaint,
          ...pixels,
        }
      } as Partial<TState>));
    },
  });
}
