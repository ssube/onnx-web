import { Maybe, doesExist, mustExist } from '@apextoaster/js-utils';
import { Delete, Replay } from '@mui/icons-material';
import { Alert, Box, Card, CardContent, IconButton, Tooltip } from '@mui/material';
import { Stack } from '@mui/system';
import { useMutation } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { FailedJobResponse, JobStatus, RetryParams, UnknownJobResponse } from '../../types/api-v2.js';

export interface ErrorCardProps {
  image: FailedJobResponse | UnknownJobResponse;
  retry: Maybe<RetryParams>;
}

export function ErrorCard(props: ErrorCardProps) {
  const { image, retry: retryParams } = props;

  const client = mustExist(useContext(ClientContext));
  const { params } = mustExist(useContext(ConfigContext));

  const state = mustExist(useContext(StateContext));
  const { pushHistory, removeHistory } = useStore(state, selectActions, shallow);
  const { t } = useTranslation();

  async function retryImage() {
    removeHistory(image);

    if (doesExist(retryParams)) {
      const { job: nextJob, retry: nextRetry } = await client.retry(retryParams);
      pushHistory(nextJob, nextRetry);
    }
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
          <Alert severity='error'>
            {t('loading.progress', image)}
            <br />
            {t(getImageErrorReason(image))}
          </Alert>
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

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    pushHistory: state.pushHistory,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeHistory: state.removeHistory,
  };
}

export const ANY_ERROR = 'error.';
export const IMAGE_ERROR = `${ANY_ERROR}image.`;
export const UNKNOWN_ERROR = `${IMAGE_ERROR}unknown`;

export function getImageErrorReason(image: FailedJobResponse | UnknownJobResponse) {
  if (image.status === JobStatus.FAILED) {
    const error = image.error;
    if (doesExist(error) && error.startsWith(ANY_ERROR)) {
      return error;
    }

    return `${IMAGE_ERROR}${error}`;
  }

  return UNKNOWN_ERROR;
}
