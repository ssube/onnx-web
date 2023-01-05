import { mustExist } from '@apextoaster/js-utils';
import * as React from 'react';
import ReactDOM from 'react-dom/client';
import { OnnxWeb } from './components/OnnxWeb';

export function main() {
  const appElement = mustExist(document.getElementById('app'));
  const app = ReactDOM.createRoot(appElement);
  app.render(<OnnxWeb />);
}

main();
