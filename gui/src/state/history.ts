import { Maybe } from '@apextoaster/js-utils';
import { ImageResponse, ReadyResponse, RetryParams } from '../types/api.js';
import { Slice } from './types.js';
import { DEFAULT_HISTORY } from '../constants.js';
import { JobResponse } from '../types/api-v2.js';

export interface HistoryItem {
  image: ImageResponse;
  ready: Maybe<ReadyResponse>;
  retry: Maybe<RetryParams>;
}

export interface HistoryItemV2 {
  image: JobResponse;
  retry: Maybe<RetryParams>;
}

export interface HistorySlice {
  history: Array<HistoryItemV2>;
  limit: number;

  pushHistory(image: JobResponse, retry?: RetryParams): void;
  removeHistory(image: JobResponse): void;

  setLimit(limit: number): void;
  setReady(image: JobResponse): void;
}

export function createHistorySlice<TState extends HistorySlice>(): Slice<TState, HistorySlice> {
  return (set) => ({
    history: [],
    limit: DEFAULT_HISTORY.limit,
    pushHistory(image, retry) {
      set((prev) => ({
        ...prev,
        history: [
          {
            image,
            ready: undefined,
            retry,
          },
          ...prev.history,
        ].slice(0, prev.limit + DEFAULT_HISTORY.scrollback),
      }));
    },
    removeHistory(image) {
      set((prev) => ({
        ...prev,
        history: prev.history.filter((it) => it.image.name !== image.name),
      }));
    },
    setLimit(limit) {
      set((prev) => ({
        ...prev,
        limit,
      }));
    },
    setReady(image) {
      set((prev) => {
        const history = [...prev.history];
        const idx = history.findIndex((it) => it.image.name === image.name);
        if (idx >= 0) {
          history[idx].image = image;
        } else {
          // TODO: error
        }

        return {
          ...prev,
          history,
        };
      });
    },
  });
}
