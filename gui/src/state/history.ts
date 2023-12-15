import { Maybe } from '@apextoaster/js-utils';
import { ImageResponse, ReadyResponse, RetryParams } from '../types/api.js';
import { Slice } from './types.js';
import { DEFAULT_HISTORY } from '../constants.js';

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
        history: prev.history.filter((it) => it.image.outputs[0].key !== image.outputs[0].key),
      }));
    },
    setLimit(limit) {
      set((prev) => ({
        ...prev,
        limit,
      }));
    },
    setReady(image, ready) {
      set((prev) => {
        const history = [...prev.history];
        const idx = history.findIndex((it) => it.image.outputs[0].key === image.outputs[0].key);
        if (idx >= 0) {
          history[idx].ready = ready;
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
