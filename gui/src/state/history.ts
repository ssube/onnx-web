import { Maybe } from '@apextoaster/js-utils';
import { ImageResponse, ReadyResponse, RetryParams } from '../types/api.js';

export interface HistoryItem {
  image: ImageResponse;
  ready: Maybe<ReadyResponse>;
  retry: Maybe<RetryParams>;
}

export interface HistorySlice {
  history: Array<HistoryItem>;
  limit: number;

  pushHistory(image: ImageResponse, retry?: RetryParams): void;
  removeHistory(image: ImageResponse): void;
  setLimit(limit: number): void;
  setReady(image: ImageResponse, ready: ReadyResponse): void;
}
