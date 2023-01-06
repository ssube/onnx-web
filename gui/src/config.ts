import { STATUS_SUCCESS } from './api/client.js';

export interface Config {
  api: {
    root: string;
  };
  default: {
    model: string;
    platform: string;
    scheduler: string;
    prompt: string;
  };
}

export async function loadConfig(): Promise<Config> {
  const configPath = new URL('./config.json', window.origin);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}
