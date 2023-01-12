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
  const state = useStore(mustExist(useContext(StateContext)));
  const { images } = state.history;

  const children = [];

  if (state.history.loading) {
    children.push(<LoadingCard key='loading' height={512} width={512} />); // TODO: get dimensions from config
  }

  function removeHistory(image: ApiResponse) {
    state.setHistory(images.filter((item) => image.output !== item.output));
  }

  if (images.length > 0) {
    children.push(...images.map((item) => <ImageCard key={item.output} value={item} onDelete={removeHistory} />));
  } else {
    if (state.history.loading === false) {
      children.push(<div>No results. Press Generate.</div>);
    }
  }

  const limited = children.slice(0, state.history.limit);

  return <Grid container spacing={2}>{limited.map((child, idx) => <Grid item key={idx} xs={6}>{child}</Grid>)}</Grid>;
}
