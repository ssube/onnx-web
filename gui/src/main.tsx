/* eslint-disable no-console */
import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { merge } from 'lodash';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';
import { satisfies } from 'semver';
import { createStore } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { makeClient } from './client.js';
import { ParamsVersionError } from './components/error/ParamsVersion.js';
import { ServerParamsError } from './components/error/ServerParams.js';
import { OnnxError } from './components/OnnxError.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import { Config, loadConfig, PARAM_VERSION } from './config.js';
import { ClientContext, ConfigContext, createStateSlices, OnnxState, StateContext } from './state.js';

export function getApiRoot(config: Config): string {
  const query = new URLSearchParams(window.location.search);
  const api = query.get('api');

  if (doesExist(api)) {
    return api;
  } else {
    return config.api.root;
  }
}

export async function main() {
  // load config from GUI server
  const config = await loadConfig();

  // use that to create an API client
  const root = getApiRoot(config);
  const client = makeClient(root);

  // prep react-dom
  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);

  try {
    // load full params from the API server and merge with the initial client config
    const params = await client.params();
    const version = mustDefault(params.version, '0.0.0');
    if (satisfies(version, PARAM_VERSION)) {
      // check version here
      merge(params, config.params);

      // prep zustand with a slice for each tab, using local storage
      const {
        createBrushSlice,
        createDefaultSlice,
        createHistorySlice,
        createImg2ImgSlice,
        createInpaintSlice,
        createModelSlice,
        createOutpaintSlice,
        createTxt2ImgSlice,
        createUpscaleSlice,
      } = createStateSlices(params);
      const state = createStore<OnnxState, [['zustand/persist', OnnxState]]>(persist((...slice) => ({
        ...createBrushSlice(...slice),
        ...createDefaultSlice(...slice),
        ...createHistorySlice(...slice),
        ...createImg2ImgSlice(...slice),
        ...createInpaintSlice(...slice),
        ...createModelSlice(...slice),
        ...createTxt2ImgSlice(...slice),
        ...createOutpaintSlice(...slice),
        ...createUpscaleSlice(...slice),
      }), {
        name: 'onnx-web',
        partialize(s) {
          return {
            ...s,
            img2img: {
              ...s.img2img,
              source: undefined,
            },
            inpaint: {
              ...s.inpaint,
              mask: undefined,
              source: undefined,
            },
            upscaleTab: {
              ...s.upscaleTab,
              source: undefined,
            },
          };
        },
        storage: createJSONStorage(() => localStorage),
        version: 3,
      }));

      // prep react-query client
      const query = new QueryClient();

      // go
      app.render(<QueryClientProvider client={query}>
        <ClientContext.Provider value={client}>
          <ConfigContext.Provider value={params}>
            <StateContext.Provider value={state}>
              <OnnxWeb />
            </StateContext.Provider>
          </ConfigContext.Provider>
        </ClientContext.Provider>
      </QueryClientProvider>);
    } else {
      app.render(<OnnxError root={root}>
        <ParamsVersionError root={root} version={version} />
      </OnnxError>);
    }
  } catch (err) {
    app.render(<OnnxError root={root}>
      <ServerParamsError root={root} error={err} />
    </OnnxError>);
  }
}

window.addEventListener('load', () => {
  console.log('launching onnx-web');
  main().catch((err) => {
    console.error('error in main', err);
  });
}, false);
