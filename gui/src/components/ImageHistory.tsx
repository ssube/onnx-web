import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Grid, Typography } from '@mui/material';
import { useContext } from 'react';
import * as React from 'react';
import { useStore } from 'zustand';

import { StateContext } from '../state.js';
import { ImageCard } from './ImageCard.js';
import { LoadingCard } from './LoadingCard.js';

export function ImageHistory() {
  const history = useStore(mustExist(useContext(StateContext)), (state) => state.history);
  const limit = useStore(mustExist(useContext(StateContext)), (state) => state.limit);
  const loading = useStore(mustExist(useContext(StateContext)), (state) => state.loading);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeHistory = useStore(mustExist(useContext(StateContext)), (state) => state.removeHistory);

  const children = [];

  if (loading.length > 0) {
    children.push(...loading.map((item) => <LoadingCard key={`loading-${item.image.outputs[0].key}`} index={0} loading={item.image} />));
  }

  if (history.length > 0) {
    children.push(...history.map((item) => <ImageCard key={`history-${item.outputs[0].key}`} value={item} onDelete={removeHistory} />));
  } else {
    if (doesExist(loading) === false) {
      children.push(<Typography>No results. Press Generate.</Typography>);
    }
  }

  const limited = children.slice(0, limit);

  return <Grid container spacing={2}>{limited.map((child, idx) => <Grid item key={idx} xs={6}>{child}</Grid>)}</Grid>;
}
