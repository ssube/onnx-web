import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Card, CardContent, CircularProgress, Typography } from '@mui/material';
import { Stack } from '@mui/system';
import { useMutation, useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useContext, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { ImageResponse } from '../../client/types.js';
import { POLL_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state.js';

const LOADING_PERCENT = 100;
const LOADING_OVERAGE = 99;

export interface LoadingCardProps {
  image: ImageResponse;
  index: number;
}

export function LoadingCard(props: LoadingCardProps) {
  const { image, index } = props;
  const { steps } = props.image.params;

  const client = mustExist(useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const store = mustExist(useContext(StateContext));
  const { removeHistory, setReady } = useStore(store, selectActions);
  const { t } = useTranslation();

  const cancel = useMutation(() => client.cancel(image.outputs[index].key));
  const ready = useQuery(['ready', image.outputs[index].key], () => client.ready(image.outputs[index].key), {
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
      return t('loading.unknown');
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
      removeHistory(props.image);
    }
  }, [cancel.status]);

  useEffect(() => {
    if (ready.status === 'success' && getReady()) {
      setReady(props.image, ready.data);
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
          <Typography>{t('loading.progress', {
            current: getProgress(),
            total: getTotal(),
          })}</Typography>
          <Button onClick={() => cancel.mutate()}>{t('loading.cancel')}</Button>
        </Stack>
      </Box>
    </CardContent>
  </Card>;
}

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeHistory: state.removeHistory,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setReady: state.setReady,
  };
}
