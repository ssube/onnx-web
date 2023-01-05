/* eslint-disable no-console */
import { mustExist } from '@apextoaster/js-utils';
import * as React from 'react';
import ReactDOM from 'react-dom/client';

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

  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  const client = makeClient(config.api.root);
  app.render(<OnnxWeb client={client} />);
}

window.addEventListener('load', () => {
  console.log('launching onnx-web');
  main().catch((err) => {
    console.error('error in main', err);
  });
}, false);
