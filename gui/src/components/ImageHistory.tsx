import { mustExist } from '@apextoaster/js-utils';
import { Grid } from '@mui/material';
import { useContext } from 'react';
import * as React from 'react';
import { useStore } from 'zustand';

import { ApiResponse } from '../api/client.js';
import { StateContext } from '../main.js';
import { ImageCard } from './ImageCard.js';
import { LoadingCard } from './LoadingCard.js';

export function ImageHistory() {
  const history = useStore(mustExist(useContext(StateContext)), (state) => state.history);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHistory = useStore(mustExist(useContext(StateContext)), (state) => state.setHistory);

  const { images } = history;

  const children = [];

  if (history.loading) {
    children.push(<LoadingCard key='loading' height={512} width={512} />); // TODO: get dimensions from config
  }

  function removeHistory(image: ApiResponse) {
    setHistory(images.filter((item) => image.output !== item.output));
  }

  if (images.length > 0) {
    children.push(...images.map((item) => <ImageCard key={item.output} value={item} onDelete={removeHistory} />));
  } else {
    if (history.loading === false) {
      children.push(<div>No results. Press Generate.</div>);
    }
  }

  const limited = children.slice(0, history.limit);

  return <Grid container spacing={2}>{limited.map((child, idx) => <Grid item key={idx} xs={6}>{child}</Grid>)}</Grid>;
}
