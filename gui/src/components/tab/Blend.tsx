import { mustDefault, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../../config.js';
import { BLEND_SOURCES, ClientContext, StateContext } from '../../state.js';
import { range } from '../../utils.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';

export function Blend() {
  async function uploadSource() {
    const { blend, blendModel, blendUpscale } = state.getState();
    const { image, retry } = await client.blend(blendModel, {
      ...blend,
      mask: mustExist(blend.mask),
      sources: mustExist(blend.sources), // TODO: show an error if this doesn't exist
    }, blendUpscale);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries(['ready']),
  });

  const state = mustExist(useContext(StateContext));
  const brush = useStore(state, (s) => s.blendBrush);
  const blend = useStore(state, (s) => s.blend);
  const upscale = useStore(state, (s) => s.blendUpscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBlend = useStore(state, (s) => s.setBlend);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBrush = useStore(state, (s) => s.setBlendBrush);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setBlendUpscale);
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
        brush={brush}
        source={sources[0]}
        mask={blend.mask}
        onSave={(mask) => {
          setBlend({
            mask,
          });
        }}
        setBrush={setBrush}
      />
      <UpscaleControl upscale={upscale} setUpscale={setUpscale} />
      <Button
        disabled={sources.length < 2}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
