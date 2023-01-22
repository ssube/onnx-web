import { doesExist, mustExist } from '@apextoaster/js-utils';
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
  const { params } = mustExist(useContext(ConfigContext));

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(mustExist(useContext(StateContext)), (state) => state.pushHistory);

  const query = useQuery('ready', () => client.ready(props.loading), {
    // data will always be ready without this, even if the API says its not
    cacheTime: 0,
    refetchInterval: POLL_TIME,
  });

  function ready() {
    return doesExist(query.data) && query.data.ready;
  }

  React.useEffect(() => {
    if (query.status === 'success' && query.data.ready) {
      pushHistory(props.loading);
    }
  }, [query.status, ready()]);

  return <Card sx={{ maxWidth: params.width.default }}>
    <CardContent sx={{ height: params.height.default }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: params.height.default,
      }}>
        <CircularProgress />
      </div>
    </CardContent>
  </Card>;
}
