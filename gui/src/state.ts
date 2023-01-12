import { ApiResponse, BaseImgParams, Img2ImgParams, InpaintParams, Txt2ImgParams } from './api/client.js';
import { ConfigState } from './config.js';

interface TabState<TabParams extends BaseImgParams> {
  params: ConfigState<Required<TabParams>>;

  reset(): void;
  update(params: Partial<ConfigState<Required<TabParams>>>): void;
}

interface OnnxState {
  defaults: {
    params: Required<BaseImgParams>;
    update(newParams: Partial<BaseImgParams>): void;
  };
  txt2img: {
    params: ConfigState<Required<Txt2ImgParams>>;

    reset(): void;
    update(newParams: Partial<ConfigState<Required<Txt2ImgParams>>>): void;
  };
  img2img: {
    params: ConfigState<Required<Img2ImgParams>>;

    reset(): void;
    update(newParams: Partial<ConfigState<Required<Img2ImgParams>>>): void;
  };
  inpaint: {
    params: ConfigState<Required<InpaintParams>>;

    reset(): void;
    update(newParams: Partial<ConfigState<Required<InpaintParams>>>): void;
  };
  history: {
    images: Array<ApiResponse>;
    limit: number;
    loading: boolean;

    setLimit(limit: number): void;
    setLoading(loading: boolean): void;
    setHistory(newHistory: Array<ApiResponse>): void;
    pushHistory(newImage: ApiResponse): void;
  };
}

