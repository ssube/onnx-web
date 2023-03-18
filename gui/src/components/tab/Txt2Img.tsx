import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { ImageControl } from '../control/ImageControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { NumericField } from '../input/NumericField.js';

export function Txt2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function generateImage() {
    const { model, txt2img, upscale } = state.getState();
    const { image, retry } = await client.txt2img(model, txt2img, upscale);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const generate = useMutation(generateImage, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const height = useStore(state, (s) => s.txt2img.height);
  const width = useStore(state, (s) => s.txt2img.width);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setTxt2Img = useStore(state, (s) => s.setTxt2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <ImageControl selector={(s) => s.txt2img} onChange={setTxt2Img} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label={t('parameter.width')}
          min={params.width.min}
          max={params.width.max}
          step={params.width.step}
          value={width}
          onChange={(value) => {
            setTxt2Img({
              width: value,
            });
          }}
        />
        <NumericField
          label={t('parameter.height')}
          min={params.height.min}
          max={params.height.max}
          step={params.height.step}
          value={height}
          onChange={(value) => {
            setTxt2Img({
              height: value,
            });
          }}
        />
      </Stack>
      <UpscaleControl />
      <Button
        variant='contained'
        onClick={() => generate.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
