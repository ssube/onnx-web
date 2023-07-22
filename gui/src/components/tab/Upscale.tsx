import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { HighresParams, ModelParams, UpscaleParams, UpscaleReqParams } from '../../client/types.js';
import { IMAGE_FILTER } from '../../config.js';
import { ClientContext, OnnxState, StateContext, TabState } from '../../state.js';
import { HighresControl } from '../control/HighresControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { PromptInput } from '../input/PromptInput.js';
import { Profiles } from '../Profiles.js';

export function Upscale() {
  async function uploadSource() {
    const { upscaleHighres, upscaleUpscale, upscaleModel, upscale } = store.getState();
    const { image, retry } = await client.upscale(upscaleModel, {
      ...upscale,
      source: mustExist(upscale.source), // TODO: show an error if this doesn't exist
    }, upscaleUpscale, upscaleHighres);

    pushHistory(image, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries([ 'ready' ]),
  });

  const store = mustExist(useContext(StateContext));
  const { pushHistory, setHighres, setModel, setParams, setUpscale } = useStore(store, selectActions, shallow);
  const model = useStore(store, selectModel);
  const params = useStore(store, selectParams);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <Profiles
        selectHighres={selectHighres}
        selectParams={selectParams}
        selectUpscale={selectUpscale}
        setParams={setParams}
        setHighres={setHighres}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} />
      <ImageInput
        filter={IMAGE_FILTER}
        image={params.source}
        label={t('input.image.source')}
        onChange={(file) => {
          setParams({
            source: file,
          });
        }}
      />
      <PromptInput
        selector={selectParams}
        onChange={(value) => {
          setParams(value);
        }}
      />
      <HighresControl selectHighres={selectHighres} setHighres={setHighres} />
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <Button
        disabled={doesExist(params.source) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    pushHistory: state.pushHistory,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setHighres: state.setUpscaleHighres,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setModel: state.setUpscaleModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setParams: state.setUpscale,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setUpscaleUpscale,
  };
}

export function selectModel(state: OnnxState): ModelParams {
  return state.upscaleModel;
}

export function selectParams(state: OnnxState): TabState<UpscaleReqParams> {
  return state.upscale;
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.upscaleHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.upscaleUpscale;
}
