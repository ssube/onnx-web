import { Maybe, doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Card, CardContent, CircularProgress, Typography } from '@mui/material';
import { Stack } from '@mui/system';
import { useMutation, useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useContext, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { POLL_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { JobResponse, JobStatus } from '../../types/api-v2.js';
import { visibleIndex } from '../../utils.js';

const LOADING_PERCENT = 100;
const LOADING_OVERAGE = 99;

export interface LoadingCardProps {
  image: JobResponse;
}

export function LoadingCard(props: LoadingCardProps) {
  const { image } = props;

  const client = mustExist(useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const store = mustExist(useContext(StateContext));
  const { removeHistory, setReady } = useStore(store, selectActions, shallow);
  const { t } = useTranslation();

  const cancel = useMutation(() => client.cancel([image.name]));
  const ready = useQuery(['ready', image.name], () => client.status([image.name]), {
    // data will always be ready without this, even if the API says its not
    cacheTime: 0,
    refetchInterval: POLL_TIME,
  });

  function renderProgress() {
    const progress = getProgress(ready.data);
    const total = getTotal(ready.data);
    if (progress > 0 && progress <= total) {
      return <CircularProgress variant='determinate' value={getPercent(progress, total)} />;
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
    if (ready.status === 'success') {
      setReady(ready.data[0]);
    }
  }, [ready.status, getStatus(ready.data), getProgress(ready.data)]);

  const status = useMemo(() => getStatus(ready.data), [ready.data]);

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
          {
            status === JobStatus.PENDING ?
              <Typography>{t('loading.queue', getQueue(ready.data))}</Typography> :
              <Typography>{t('loading.progress', selectStatus(ready.data, image))}</Typography>
          }
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

export function selectStatus(data: Maybe<Array<JobResponse>>, defaultData: JobResponse) {
  if (doesExist(data) && data.length > 0) {
    return {
      steps: data[0].steps,
      stages: data[0].stages,
      tiles: data[0].tiles,
    };
  }

  return {
    steps: defaultData.steps,
    stages: defaultData.stages,
    tiles: defaultData.tiles,
  };
}

export function getPercent(current: number, total: number): number {
  if (current > total) {
    // steps was not complete, show 99% until done
    return LOADING_OVERAGE;
  }

  const pct = current / total;
  return Math.ceil(pct * LOADING_PERCENT);
}

export function getProgress(data: Maybe<Array<JobResponse>>) {
  if (doesExist(data)) {
    return data[0].steps.current;
  }

  return 0;
}

export function getTotal(data: Maybe<Array<JobResponse>>) {
  if (doesExist(data)) {
    return data[0].steps.total;
  }

  return 0;
}

function getStatus(data: Maybe<Array<JobResponse>>) {
  if (doesExist(data)) {
    return data[0].status;
  }

  return JobStatus.PENDING;
}

function getQueue(data: Maybe<Array<JobResponse>>) {
  if (doesExist(data) && data[0].status === JobStatus.PENDING) {
    const { current, total } = data[0].queue;
    return {
      current: visibleIndex(current),
      total: total.toFixed(0),
    };
  }

  return {
    current: '0',
    total: '0',
  };
}
