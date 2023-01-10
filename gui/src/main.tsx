/* eslint-disable no-console */
import { mustExist } from '@apextoaster/js-utils';
import { merge } from 'lodash';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from 'react-query';

import { makeClient } from './api/client.js';
import { OnnxWeb } from './components/OnnxWeb.js';
import { loadConfig } from './config.js';

export async function main() {
  const config = await loadConfig();
  const client = makeClient(config.api.root);
  const params = await client.params();
  merge(params, config.params);

  const query = new QueryClient();
  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  app.render(<QueryClientProvider client={query}>
    <OnnxWeb client={client} config={params} />
  </QueryClientProvider>);
}

window.addEventListener('load', () => {
  console.log('launching onnx-web');
  main().catch((err) => {
    console.error('error in main', err);
  });
}, false);
