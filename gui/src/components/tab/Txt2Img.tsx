import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { NumericField } from '../input/NumericField.js';
import { ModelControl } from '../control/ModelControl.js';
import { Profiles } from '../Profiles.js';

export function Txt2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function generateImage() {
    const { image, retry } = await client.txt2img(model, txt2img, upscale, highres);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const generate = useMutation(generateImage, {
    onSuccess: () => query.invalidateQueries([ 'ready' ]),
  });

  const state = mustExist(useContext(StateContext));
  const txt2img = useStore(state, (s) => s.txt2img);
  const model = useStore(state, (s) => s.txt2imgModel);
  const highres = useStore(state, (s) => s.txt2imgHighres);
  const upscale = useStore(state, (s) => s.txt2imgUpscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setParams = useStore(state, (s) => s.setTxt2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHighres = useStore(state, (s) => s.setTxt2ImgHighres);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setTxt2ImgUpscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setModel = useStore(state, (s) => s.setTxt2ImgModel);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <Profiles params={txt2img} setParams={setParams} highres={highres} setHighres={setHighres} upscale={upscale} setUpscale={setUpscale} />
      <ModelControl model={model} setModel={setModel} />
      <ImageControl selector={(s) => s.txt2img} onChange={setParams} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label={t('parameter.width')}
          min={params.width.min}
          max={params.width.max}
          step={params.width.step}
          value={txt2img.width}
          onChange={(value) => {
            setParams({
              width: value,
            });
          }}
        />
        <NumericField
          label={t('parameter.height')}
          min={params.height.min}
          max={params.height.max}
          step={params.height.step}
          value={txt2img.height}
          onChange={(value) => {
            setParams({
              height: value,
            });
          }}
        />
      </Stack>
      <HighresControl highres={highres} setHighres={setHighres} />
      <UpscaleControl upscale={upscale} setUpscale={setUpscale} />
      <Button
        variant='contained'
        onClick={() => generate.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
