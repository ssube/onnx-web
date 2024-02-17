import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { IMAGE_FILTER, STALE_TIME, STANDARD_SPACING } from '../../constants.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { TabState } from '../../state/types.js';
import { JobType } from '../../types/api-v2.js';
import { ExperimentalParams, HighresParams, Img2ImgParams, ModelParams, UpscaleParams } from '../../types/params.js';
import { Profiles } from '../Profiles.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';
import { ExperimentalControl } from '../control/ExperimentalControl.js';

export function Img2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function uploadSource() {
    const state = store.getState();
    const img2img = selectParams(state);

    const { job, retry } = await client.img2img(model, {
      ...img2img,
      source: mustExist(img2img.source), // TODO: show an error if this doesn't exist
    }, selectUpscale(state), selectHighres(state));

    pushHistory(job, retry);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries(['ready']),
  });

  const filters = useQuery(['filters'], async () => client.filters(), {
    staleTime: STALE_TIME,
  });
  const models = useQuery(['models'], async () => client.models(), {
    staleTime: STALE_TIME,
  });

  const store = mustExist(useContext(StateContext));
  const { pushHistory, setHighres, setImg2Img, setModel, setUpscale, setExperimental } = useStore(store, selectActions, shallow);
  const { loopback, source, sourceFilter, strength } = useStore(store, selectReactParams, shallow);
  const model = useStore(store, selectModel);

  const { t } = useTranslation();

  return <Box>
    <Stack spacing={STANDARD_SPACING}>
      <Profiles
        selectHighres={selectHighres}
        selectModel={selectModel}
        selectParams={selectParams}
        selectUpscale={selectUpscale}
        setHighres={setHighres}
        setModel={setModel}
        setParams={setImg2Img}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} tab={JobType.IMG2IMG} />
      <ImageInput
        filter={IMAGE_FILTER}
        image={source}
        label={t('input.image.source')}
        onChange={(file) => {
          setImg2Img({
            source: file,
          });
        }}
      />
      <ImageControl selector={selectParams} onChange={setImg2Img} />
      <Stack direction='row' spacing={STANDARD_SPACING}>
        <QueryList
          id='control'
          labelKey='model.control'
          name={t('modelType.control')}
          query={{
            result: models,
            selector: (result) => result.networks.filter((network) => network.type === 'control').map((network) => network.name),
          }}
          value={model.control}
          onChange={(newControl) => {
            setModel({
              control: newControl,
            });
          }}
        />
        <QueryList
          id='sources'
          labelKey={'sourceFilter'}
          name={t('parameter.sourceFilter')}
          query={{
            result: filters,
            selector: (f) => f.source,
          }}
          showNone
          value={sourceFilter}
          onChange={(newFilter) => {
            setImg2Img({
              sourceFilter: newFilter,
            });
          }}
        />
        <NumericField
          decimal
          label={t('parameter.strength')}
          min={params.strength.min}
          max={params.strength.max}
          step={params.strength.step}
          value={strength}
          onChange={(value) => {
            setImg2Img({
              strength: value,
            });
          }}
        />
        <NumericField
          label={t('parameter.loopback')}
          min={params.loopback.min}
          max={params.loopback.max}
          step={params.loopback.step}
          value={loopback}
          onChange={(value) => {
            setImg2Img({
              loopback: value,
            });
          }}
        />
      </Stack>
      <HighresControl selectHighres={selectHighres} setHighres={setHighres} />
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <ExperimentalControl selectExperimental={selectExperimental} setExperimental={setExperimental} />
      <Button
        disabled={doesExist(source) === false}
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
    setImg2Img: state.setImg2Img,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setHighres: state.setImg2ImgHighres,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setModel: state.setImg2ImgModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setImg2ImgUpscale,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setExperimental: state.setImg2ImgExperimental,
  };
}

export function selectModel(state: OnnxState): ModelParams {
  return state.img2imgModel;
}

export function selectParams(state: OnnxState): TabState<Img2ImgParams> {
  return state.img2img;
}

export function selectReactParams(state: OnnxState) {
  return {
    loopback: state.img2img.loopback,
    source: state.img2img.source,
    sourceFilter: state.img2img.sourceFilter,
    strength: state.img2img.strength,
  };
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.img2imgHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.img2imgUpscale;
}

export function selectExperimental(state: OnnxState): ExperimentalParams {
  return state.img2imgExperimental;
}
