import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from '../../client/types.js';
import { ClientContext, ConfigContext, OnnxState, StateContext, TabState } from '../../state.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { NumericField } from '../input/NumericField.js';
import { Profiles } from '../Profiles.js';

export function Txt2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function generateImage() {
    const innerState = state.getState();
    const { image, retry } = await client.txt2img(model, selectParams(innerState), selectUpscale(innerState), selectHighres(innerState));

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const generate = useMutation(generateImage, {
    onSuccess: () => query.invalidateQueries([ 'ready' ]),
  });

  const state = mustExist(useContext(StateContext));
  const height = useStore(state, (s) => s.txt2img.height);
  const width = useStore(state, (s) => s.txt2img.width);
  const model = useStore(state, selectModel);
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
      <Profiles
        selectParams={selectParams}
        selectHighres={selectHighres}
        selectUpscale={selectUpscale}
        setParams={setParams}
        setHighres={setHighres}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} />
      <ImageControl selector={selectParams} onChange={setParams} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label={t('parameter.width')}
          min={params.width.min}
          max={params.width.max}
          step={params.width.step}
          value={width}
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
          value={height}
          onChange={(value) => {
            setParams({
              height: value,
            });
          }}
        />
      </Stack>
      <HighresControl selectHighres={selectHighres} setHighres={setHighres} />
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <Button
        variant='contained'
        onClick={() => generate.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}

export function selectModel(state: OnnxState): ModelParams {
  return state.txt2imgModel;
}

export function selectParams(state: OnnxState): TabState<Txt2ImgParams> {
  return state.txt2img;
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.txt2imgHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.txt2imgUpscale;
}
