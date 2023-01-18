import { mustExist } from '@apextoaster/js-utils';
import { Card, CardContent, CircularProgress } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useQuery } from 'react-query';
import { useStore } from 'zustand';

import { ImageResponse } from '../client.js';
import { POLL_TIME } from '../config.js';
import { ClientContext, ConfigContext, StateContext } from '../state.js';

export interface LoadingCardProps {
  loading: ImageResponse;
}

export function LoadingCard(props: LoadingCardProps) {
  const client = mustExist(React.useContext(ClientContext));
  const config = mustExist(useContext(ConfigContext));

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(mustExist(useContext(StateContext)), (state) => state.pushHistory);

  const ready = useQuery('ready', () => client.ready(props.loading), {
    // data will always be ready without this, even if the API says its not
    cacheTime: 0,
    refetchInterval: POLL_TIME,
  });

  React.useEffect(() => {
    if (ready.status === 'success' && ready.data.ready) {
      pushHistory(props.loading);
    }
  }, [ready.status, ready.data?.ready]);

  return <Card sx={{ maxWidth: config.width.default }}>
    <CardContent sx={{ height: config.height.default }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: config.height.default,
      }}>
        <CircularProgress />
      </div>
    </CardContent>
  </Card>;
}
