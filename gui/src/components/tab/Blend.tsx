import { mustDefault, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../../config.js';
import { BLEND_SOURCES, ClientContext, StateContext } from '../../state.js';
import { range } from '../../utils.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';

export function Blend() {
  async function uploadSource() {
    const { model, blend, upscale } = state.getState();
    const { image, retry } = await client.blend(model, {
      ...blend,
      mask: mustExist(blend.mask),
      sources: mustExist(blend.sources), // TODO: show an error if this doesn't exist
    }, upscale);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const blend = useStore(state, (s) => s.blend);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBlend = useStore(state, (s) => s.setBlend);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  const sources = mustDefault(blend.sources, []);

  return <Box>
    <Stack spacing={2}>
      {range(BLEND_SOURCES).map((idx) =>
        <ImageInput
          key={`source-${idx.toFixed(0)}`}
          filter={IMAGE_FILTER}
          image={sources[idx]}
          hideSelection={true}
          label={t('input.image.source')}
          onChange={(file) => {
            const newSources = [...sources];
            newSources[idx] = file;

            setBlend({
              sources: newSources,
            });
          }}
        />
      )}
      <MaskCanvas
        source={sources[0]}
        mask={blend.mask}
        onSave={(mask) => {
          setBlend({
            mask,
          });
        }}
      />
      <UpscaleControl />
      <Button
        disabled={sources.length === 0}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
