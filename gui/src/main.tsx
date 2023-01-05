import { mustExist } from '@apextoaster/js-utils';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { makeClient } from './api/client';
import { OnnxWeb } from './components/OnnxWeb';
import { CONFIG } from './config';

export function main() {
  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  const client = makeClient(CONFIG.api.root);
  app.render(<OnnxWeb client={client} />);
}

main();
