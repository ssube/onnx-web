import { mustDefault, mustExist, timeout, TimeoutError } from '@apextoaster/js-utils';
import { Box, CircularProgress } from '@mui/material';
import { createLogger, Logger } from 'browser-bunyan';
import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { QueryClient, QueryClientProvider } from 'react-query';
import { satisfies } from 'semver';
import { createStore } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { ApiClient, makeClient } from './client/api.js';
import { LOCAL_CLIENT } from './client/local.js';
import { ParamsVersionError } from './components/error/ParamsVersion.js';
import { ServerParamsError } from './components/error/ServerParams.js';
import { LoadingScreen } from './components/LoadingScreen.js';
import { OnnxError } from './components/OnnxError.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import {
  Config,
  getApiRoot,
  isDebug,
  loadConfig,
  mergeConfig,
  PARAM_VERSION,
  ServerParams,
} from './config.js';
import {
  ClientContext,
  ConfigContext,
  createStateSlices,
  LoggerContext,
  OnnxState,
  STATE_KEY,
  STATE_VERSION,
  StateContext,
} from './state.js';
import { I18N_STRINGS } from './strings/all.js';

export const INITIAL_LOAD_TIMEOUT = 5_000;

export async function renderApp(config: Config, params: ServerParams, logger: Logger, client: ApiClient) {
  const completeConfig = mergeConfig(config, params);

  // prep i18next
  await i18n
    .use(LanguageDetector)
    .use(initReactI18next)
    .init({
      debug: true,
      fallbackLng: 'en',
      interpolation: {
        escapeValue: false, // not needed for react as it escapes by default
      },
      resources: I18N_STRINGS,
      returnEmptyString: false,
    });

  logger.info('getting strings from server');
  const strings = await client.strings();
  for (const [lang, translation] of Object.entries(strings)) {
    logger.debug({ lang, translation }, 'adding server strings');
    for (const [namespace, data] of Object.entries(translation)) {
      i18n.addResourceBundle(lang, namespace, data, true);
    }
  }

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

  // prep react-query client
  const query = new QueryClient();

  const reactLogger = logger.child({
    system: 'react',
  });

  // go
  return <QueryClientProvider client={query}>
    <ClientContext.Provider value={client}>
      <ConfigContext.Provider value={completeConfig}>
        <LoggerContext.Provider value={reactLogger}>
          <I18nextProvider i18n={i18n}>
            <StateContext.Provider value={state}>
              <OnnxWeb />
            </StateContext.Provider>
          </I18nextProvider>
        </LoggerContext.Provider>
      </ConfigContext.Provider>
    </ClientContext.Provider>
  </QueryClientProvider>;
}

export async function main() {
  const debug = isDebug();
  const logger = createLogger({
    name: 'onnx-web',
    level: debug ? 'debug' : 'info',
  });

  // load config from GUI server
  const config = await loadConfig();

  // use that to create an API client
  const root = getApiRoot(config);
  const client = makeClient(root);

  // prep react-dom
  const appElement = mustExist(document.getElementById('app'));
  const app = createRoot(appElement);

  try {
    logger.info('getting image parameters from server');
    app.render(<LoadingScreen />);

    // load full params from the API server and merge with the initial client config
    const params = await timeout(INITIAL_LOAD_TIMEOUT, client.params());
    const version = mustDefault(params.version, '0.0.0');
    if (satisfies(version, PARAM_VERSION)) {
      app.render(await renderApp(config, params, logger, client));
    } else {
      app.render(<OnnxError root={root}>
        <ParamsVersionError root={root} version={version} />
      </OnnxError>);
    }
  } catch (err) {
    if (err instanceof TimeoutError || (err instanceof Error && err.message.includes('Failed to fetch'))) {
      // params timed out, attempt to render without a server
      app.render(await renderApp(config, config.params as ServerParams, logger, LOCAL_CLIENT));
    } else {
      app.render(<OnnxError root={root}>
        <ServerParamsError root={root} error={err} />
      </OnnxError>);
    }
  }
}

window.addEventListener('load', () => {
  // eslint-disable-next-line no-console
  console.log('launching onnx-web');
  main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error('error in main', err);
  });
}, false);
