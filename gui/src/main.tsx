/* eslint-disable no-console */
import { Maybe, mustExist } from '@apextoaster/js-utils';
import { merge } from 'lodash';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';
import { createStore, StoreApi } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { ApiClient, makeClient } from './api/client.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import { loadConfig } from './config.js';
import { createStateSlices, OnnxState } from './state.js';

const { createContext } = React;

export async function main() {
  const config = await loadConfig();
  const client = makeClient(config.api.root);
  const params = await client.params();
  merge(params, config.params);

  const { createDefaultSlice, createHistorySlice, createImg2ImgSlice, createInpaintSlice, createTxt2ImgSlice } = createStateSlices(params);
  const state = createStore<OnnxState, [['zustand/persist', OnnxState]]>(persist((...slice) => ({
    ...createTxt2ImgSlice(...slice),
    ...createImg2ImgSlice(...slice),
    ...createInpaintSlice(...slice),
    ...createHistorySlice(...slice),
    ...createDefaultSlice(...slice),
  }), {
    name: 'onnx-web',
    partialize: (oldState) => ({
      ...oldState,
      history: {
        ...oldState.history,
        loading: false,
      },
    }),
    storage: createJSONStorage(() => localStorage),
    version: 1,
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
