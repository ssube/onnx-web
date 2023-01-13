import { mustExist } from '@apextoaster/js-utils';
import { Card, CardContent, CircularProgress } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useQuery } from 'react-query';
import { useStore } from 'zustand';

import { ApiResponse } from '../api/client.js';
import { POLL_TIME } from '../config.js';
import { ClientContext, StateContext } from '../state.js';

export interface LoadingCardProps {
  loading: ApiResponse;
}

export function LoadingCard(props: LoadingCardProps) {
  const client = mustExist(React.useContext(ClientContext));

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(mustExist(useContext(StateContext)), (state) => state.pushHistory);

  const ready = useQuery('ready', () => client.ready(props.loading), {
    refetchInterval: POLL_TIME,
  });

  React.useEffect(() => {
    if (ready.status === 'success' && ready.data.ready) {
      pushHistory(props.loading);
    }
  }, [ready.status, ready.data?.ready]);

  return <Card sx={{ maxWidth: props.loading.params.width }}>
    <CardContent sx={{ height: props.loading.params.height }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: props.loading.params.height }}>
        <CircularProgress />
      </div>
    </CardContent>
  </Card>;
}
