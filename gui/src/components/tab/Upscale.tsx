import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../../config.js';
import { ClientContext, StateContext } from '../../state.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { PromptInput } from '../input/PromptInput.js';

export function Upscale() {
  async function uploadSource() {
    const { model, upscale } = state.getState();
    const { image, retry } = await client.upscale(model, {
      ...params,
      source: mustExist(params.source), // TODO: show an error if this doesn't exist
    }, upscale);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.upscaleTab);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setSource = useStore(state, (s) => s.setUpscaleTab);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <ImageInput
        filter={IMAGE_FILTER}
        image={params.source}
        label={t('input.image.source')}
        onChange={(file) => {
          setSource({
            source: file,
          });
        }}
      />
      <PromptInput
        prompt={params.prompt}
        negativePrompt={params.negativePrompt}
        onChange={(value) => {
          setSource(value);
        }}
      />
      <UpscaleControl />
      <Button
        disabled={doesExist(params.source) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
