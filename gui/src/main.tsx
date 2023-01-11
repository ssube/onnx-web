/* eslint-disable no-console */
import { Maybe, mustExist } from '@apextoaster/js-utils';
import { merge } from 'lodash';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';
import { createStore, StoreApi } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { ApiClient, BaseImgParams, Img2ImgParams, InpaintParams, makeClient, paramsFromConfig, Txt2ImgParams } from './api/client.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import { ConfigState, loadConfig } from './config.js';

const { createContext } = React;

interface OnnxState {
  defaults: Required<BaseImgParams>;
  txt2img: ConfigState<Required<Txt2ImgParams>>;
  img2img: ConfigState<Required<Img2ImgParams>>;
  inpaint: ConfigState<Required<InpaintParams>>;

  setDefaults(newParams: Partial<BaseImgParams>): void;
  setTxt2Img(newParams: Partial<ConfigState<Required<Txt2ImgParams>>>): void;
  setImg2Img(newParams: Partial<ConfigState<Required<Img2ImgParams>>>): void;
  setInpaint(newParams: Partial<ConfigState<Required<InpaintParams>>>): void;

  resetTxt2Img(): void;
  resetImg2Img(): void;
  resetInpaint(): void;
}

export async function main() {
  const config = await loadConfig();
  const client = makeClient(config.api.root);
  const params = await client.params();
  merge(params, config.params);

  const defaults = paramsFromConfig(params);
  const state = createStore<OnnxState, [['zustand/persist', never]]>(persist((set) => ({
    defaults,
    txt2img: {
      ...defaults,
      height: params.height.default,
      width: params.width.default,
    },
    img2img: {
      ...defaults,
      strength: params.strength.default,
    },
    inpaint: {
      ...defaults,
    },
    setDefaults(newParams) {
      set((oldState) => ({
        ...oldState,
        defaults: {
          ...oldState.defaults,
          ...newParams,
        },
      }));
    },
    setTxt2Img(newParams) {
      set((oldState) => ({
        ...oldState,
        txt2img: {
          ...oldState.txt2img,
          ...newParams,
        },
      }));
    },
    setImg2Img(newParams) {
      set((oldState) => ({
        ...oldState,
        img2img: {
          ...oldState.img2img,
          ...newParams,
        },
      }));
    },
    setInpaint(newParams) {
      set((oldState) => ({
        ...oldState,
        inpaint: {
          ...oldState.inpaint,
          ...newParams,
        },
      }));
    },
    resetTxt2Img() {
      set((oldState) => ({
        ...oldState,
        txt2img: {
          ...defaults,
          height: params.height.default,
          width: params.width.default,
        },
      }));
    },
    resetImg2Img() {
      set((oldState) => ({
        ...oldState,
        img2img: {
          ...defaults,
          strength: params.strength.default,
        },
      }));
    },
    resetInpaint() {
      set((oldState) => ({
        ...oldState,
        inpaint: {
          ...defaults,
        },
      }));
    },
  }), {
    name: 'onnx-web',
    storage: createJSONStorage(() => localStorage),
  }));

  const query = new QueryClient();
  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  app.render(<QueryClientProvider client={query}>
    <ClientContext.Provider value={client}>
      <StateContext.Provider value={state}>
        <OnnxWeb client={client} config={params} />
      </StateContext.Provider>
    </ClientContext.Provider>
  </QueryClientProvider>);
}

window.addEventListener('load', () => {
  console.log('launching onnx-web');
  main().catch((err) => {
    console.error('error in main', err);
  });
}, false);

export const ClientContext = createContext<Maybe<ApiClient>>(undefined);
export const StateContext = createContext<Maybe<StoreApi<OnnxState>>>(undefined);
