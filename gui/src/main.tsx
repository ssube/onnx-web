/* eslint-disable no-console */
import { mustExist } from '@apextoaster/js-utils';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';

import { makeClient, STATUS_SUCCESS } from './api/client.js';
import { OnnxWeb } from './components/OnnxWeb.js';

export interface Config {
  api: {
    root: string;
  };
}

export async function loadConfig() {
  const configPath = new URL('./config.json', window.origin);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}

export async function main() {
  const config = await loadConfig();
  const client = makeClient(config.api.root);
  const query = new QueryClient();

  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  app.render(<QueryClientProvider client={query}><OnnxWeb client={client} /></QueryClientProvider>);
}

window.addEventListener('load', () => {
  console.log('launching onnx-web');
  main().catch((err) => {
    console.error('error in main', err);
  });
}, false);
