import { mustExist } from '@apextoaster/js-utils';
import { Delete, Replay } from '@mui/icons-material';
import { Alert, Box, Card, CardContent, IconButton, Tooltip } from '@mui/material';
import { Stack } from '@mui/system';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { ImageResponse, ReadyResponse, RetryParams } from '../../client/api.js';
import { ClientContext, ConfigContext, StateContext } from '../../state.js';

export interface ErrorCardProps {
  image: ImageResponse;
  ready: ReadyResponse;
  retry: RetryParams;
}

export function ErrorCard(props: ErrorCardProps) {
  const { image, ready, retry: retryParams } = props;

  const client = mustExist(React.useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const state = mustExist(useContext(StateContext));
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeHistory = useStore(state, (s) => s.removeHistory);
  const { t } = useTranslation();

  async function retryImage() {
    removeHistory(image);
    const { image: nextImage, retry: nextRetry } = await client.retry(retryParams);
    pushHistory(nextImage, nextRetry);
  }

  const retry = useMutation(retryImage);

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
          <Alert severity='error'>{t('loading.progress', {
            current: ready.progress,
            total: image.params.steps,
          })}</Alert>
          <Stack direction='row' spacing={2}>
            <Tooltip title={t('tooltip.retry')}>
              <IconButton onClick={() => retry.mutate()}>
                <Replay />
              </IconButton>
            </Tooltip>
            <Tooltip title={t('tooltip.delete')}>
              <IconButton onClick={() => removeHistory(image)}>
                <Delete />
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>
      </Box>
    </CardContent>
  </Card>;
}
