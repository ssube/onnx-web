import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Card, CardContent, CircularProgress, Typography } from '@mui/material';
import { Stack } from '@mui/system';
import * as React from 'react';
import { useContext, useEffect } from 'react';
import { useMutation, useQuery } from 'react-query';
import { useStore } from 'zustand';

import { ImageResponse } from '../client.js';
import { POLL_TIME } from '../config.js';
import { ClientContext, ConfigContext, StateContext } from '../state.js';

const LOADING_PERCENT = 100;
const LOADING_OVERAGE = 99;

export interface LoadingCardProps {
  loading: ImageResponse;
}

export function LoadingCard(props: LoadingCardProps) {
  const { steps } = props.loading.params;

  const client = mustExist(React.useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const state = mustExist(useContext(StateContext));
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const clearLoading = useStore(state, (s) => s.clearLoading);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setReady = useStore(state, (s) => s.setReady);

  const cancel = useMutation(() => client.cancel(props.loading));
  const ready = useQuery(`ready-${props.loading.output.key}`, () => client.ready(props.loading), {
    // data will always be ready without this, even if the API says its not
    cacheTime: 0,
    refetchInterval: POLL_TIME,
  });

  function getProgress() {
    if (doesExist(ready.data)) {
      return ready.data.progress;
    }

    return 0;
  }

  function getPercent() {
    const progress = getProgress();
    if (progress > steps) {
      // steps was not complete, show 99% until done
      return LOADING_OVERAGE;
    }

    const pct = progress / steps;
    return Math.ceil(pct * LOADING_PERCENT);
  }

  function getTotal() {
    const progress = getProgress();
    if (progress > steps) {
      // steps was not complete, show 99% until done
      return 'many';
    }

    return steps.toFixed(0);
  }

  function getReady() {
    return doesExist(ready.data) && ready.data.ready;
  }

  function renderProgress() {
    const progress = getProgress();
    if (progress > 0 && progress <= steps) {
      return <CircularProgress variant='determinate' value={getPercent()} />;
    } else {
      return <CircularProgress />;
    }
  }

  useEffect(() => {
    if (cancel.status === 'success') {
      clearLoading(props.loading);
    }
  }, [cancel.status]);

  useEffect(() => {
    if (ready.status === 'success') {
      if (ready.data.ready) {
        pushHistory(props.loading);
      } else {
        setReady(props.loading, ready.data);
      }
    }
  }, [ready.status, getReady(), getProgress()]);

  return <Card sx={{ maxWidth: params.width.default }}>
    <CardContent sx={{ height: params.height.default }}>
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: params.height.default,
      }}>
        <Stack
          direction='column'
          spacing={2}
          sx={{ alignItems: 'center' }}
        >
          {renderProgress()}
          <Typography>{getProgress()}/{getTotal()} steps</Typography>
          <Button onClick={() => cancel.mutate()}>Cancel</Button>
        </Stack>
      </Box>
    </CardContent>
  </Card>;
}
