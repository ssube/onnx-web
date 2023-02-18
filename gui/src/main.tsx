/* eslint-disable no-console */
import { mustDefault, mustExist, timeout } from '@apextoaster/js-utils';
import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';
import { satisfies } from 'semver';
import { createStore } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { createLogger } from 'browser-bunyan';

import { makeClient } from './client.js';
import { ParamsVersionError } from './components/error/ParamsVersion.js';
import { ServerParamsError } from './components/error/ServerParams.js';
import { OnnxError } from './components/OnnxError.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import { getApiRoot, loadConfig, mergeConfig, PARAM_VERSION } from './config.js';
import { ClientContext, ConfigContext, createStateSlices, OnnxState, STATE_VERSION, StateContext, LoggerContext, STATE_KEY } from './state.js';

export const INITIAL_LOAD_TIMEOUT = 5_000;

export async function main() {
  // load config from GUI server
  const config = await loadConfig();

  // use that to create an API client
  const root = getApiRoot(config);
  const client = makeClient(root);

  // prep react-dom
  const appElement = mustExist(document.getElementById('app'));
  const app = createRoot(appElement);

  try {
    // load full params from the API server and merge with the initial client config
    const params = await timeout(INITIAL_LOAD_TIMEOUT, client.params());
    const version = mustDefault(params.version, '0.0.0');
    if (satisfies(version, PARAM_VERSION)) {
      const completeConfig = mergeConfig(config, params);

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
        createBlendSlice,
        createResetSlice,
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
        ...createBlendSlice(...slice),
        ...createResetSlice(...slice),
      }), {
        name: STATE_KEY,
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
            blend: {
              ...s.blend,
              mask: undefined,
              sources: [],
            }
          };
        },
        storage: createJSONStorage(() => localStorage),
        version: STATE_VERSION,
      }));

      const logger = createLogger({
        name: 'onnx-web',
        system: 'react',
        level: 'debug',
      });

      // prep react-query client
      const query = new QueryClient();

      // go
      app.render(<QueryClientProvider client={query}>
        <ClientContext.Provider value={client}>
          <ConfigContext.Provider value={completeConfig}>
            <LoggerContext.Provider value={logger}>
              <StateContext.Provider value={state}>
                <OnnxWeb />
              </StateContext.Provider>
            </LoggerContext.Provider>
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
