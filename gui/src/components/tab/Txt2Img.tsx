import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { PipelineGrid, makeTxt2ImgGridPipeline } from '../../client/utils.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { TabState } from '../../state/types.js';
import { HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from '../../types/params.js';
import { Profiles } from '../Profiles.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { VariableControl } from '../control/VariableControl.js';
import { NumericField } from '../input/NumericField.js';

export function SizeControl() {
  const { params } = mustExist(useContext(ConfigContext));

  const store = mustExist(useContext(StateContext));
  const { height, width } = useStore(store, selectSize, shallow);
  const { setParams } = useStore(store, selectActions, shallow);

  const { t } = useTranslation();

  return <Stack direction='row' spacing={4}>
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
  </Stack>;
}

export function Txt2Img() {
  async function generateImage() {
    const state = store.getState();
    const grid = selectVariable(state);
    const params2 = selectParams(state);
    const upscale = selectUpscale(state);
    const highres = selectHighres(state);

    if (grid.enabled) {
      const chain = makeTxt2ImgGridPipeline(grid, model, params2, upscale, highres);
      const image = await client.chain(model, chain);
      pushHistory(image);
    } else {
      const { image, retry } = await client.txt2img(model, params2, upscale, highres);
      pushHistory(image, retry);
    }
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const generate = useMutation(generateImage, {
    onSuccess: () => query.invalidateQueries([ 'ready' ]),
  });

  const store = mustExist(useContext(StateContext));
  const { pushHistory, setHighres, setModel, setParams, setUpscale, setVariable } = useStore(store, selectActions, shallow);
  const model = useStore(store, selectModel);

  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <Profiles
        selectHighres={selectHighres}
        selectModel={selectModel}
        selectParams={selectParams}
        selectUpscale={selectUpscale}
        setHighres={setHighres}
        setModel={setModel}
        setParams={setParams}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} />
      <ImageControl selector={selectParams} onChange={setParams} />
      <SizeControl />
      <HighresControl selectHighres={selectHighres} setHighres={setHighres} />
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <VariableControl selectGrid={selectVariable} setGrid={setVariable} />
      <Button
        variant='contained'
        onClick={() => generate.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    pushHistory: state.pushHistory,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setHighres: state.setTxt2ImgHighres,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setModel: state.setTxt2ImgModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setParams: state.setTxt2Img,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setTxt2ImgUpscale,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setVariable: state.setTxt2ImgVariable,
  };
}

export function selectModel(state: OnnxState): ModelParams {
  return state.txt2imgModel;
}

export function selectParams(state: OnnxState): TabState<Txt2ImgParams> {
  return state.txt2img;
}

export function selectSize(state: OnnxState) {
  return {
    height: state.txt2img.height,
    width: state.txt2img.width,
  };
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.txt2imgHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.txt2imgUpscale;
}

export function selectVariable(state: OnnxState): PipelineGrid {
  return state.txt2imgVariable;
}
