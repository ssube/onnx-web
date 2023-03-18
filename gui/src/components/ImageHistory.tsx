import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Grid, Typography } from '@mui/material';
import { useContext, ReactNode } from 'react';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { StateContext } from '../state.js';
import { ImageCard } from './card/ImageCard.js';
import { LoadingCard } from './card/LoadingCard.js';
import { ErrorCard } from './card/RetryCard.js';

export function ImageHistory() {
  const history = useStore(mustExist(useContext(StateContext)), (state) => state.history);
  const limit = useStore(mustExist(useContext(StateContext)), (state) => state.limit);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeHistory = useStore(mustExist(useContext(StateContext)), (state) => state.removeHistory);

  const { t } = useTranslation();

  const children: Array<[string, ReactNode]> = [];

  if (history.length === 0) {
    children.push(['empty', <Typography>{t('history.empty')}</Typography>]);
  }

  const limited = history.slice(0, limit);
  for (const item of limited) {
    const key = item.image.outputs[0].key;

    if (doesExist(item.ready) && item.ready.ready) {
      if (item.ready.cancelled || item.ready.failed) {
        children.push([key, <ErrorCard key={`history-${key}`} image={item.image} ready={item.ready} retry={item.retry} />]);
        continue;
      }

      children.push([key, <ImageCard key={`history-${key}`} image={item.image} onDelete={removeHistory} />]);
      continue;
    }

    children.push([key, <LoadingCard key={`history-${key}`} index={0} image={item.image} />]);
  }

  return <Grid container spacing={2}>{children.map(([key, child]) => <Grid item key={key} xs={6}>{child}</Grid>)}</Grid>;
}
