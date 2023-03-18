import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Card, CardContent, Typography } from '@mui/material';
import { Stack } from '@mui/system';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation } from 'react-query';
import { useStore } from 'zustand';

import { ImageResponse, ReadyResponse } from '../../client/api.js';
import { ClientContext, ConfigContext, StateContext } from '../../state.js';

export interface ErrorCardProps {
  image: ImageResponse;
  ready: ReadyResponse;
}

export function ErrorCard(props: ErrorCardProps) {
  const { image, ready } = props;

  const client = mustExist(React.useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const state = mustExist(useContext(StateContext));
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeHistory = useStore(state, (s) => s.removeHistory);
  const { t } = useTranslation();

  // TODO: actually retry
  const retry = useMutation(() => {
    // eslint-disable-next-line no-console
    console.log('retry', image);
    return Promise.resolve(true);
  });

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
          <Typography>{t('loading.progress', {
            current: ready.progress,
            total: image.params.steps,
          })}</Typography>
          <Button onClick={() => retry.mutate()}>{t('loading.retry')}</Button>
          <Button onClick={() => removeHistory(image)}>{t('loading.remove')}</Button>
        </Stack>
      </Box>
    </CardContent>
  </Card>;
}
