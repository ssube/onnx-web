import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER, STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { ImageControl } from '../control/ImageControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';
import { HighresControl } from '../control/HighresControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { Profiles } from '../Profiles.js';

export function Img2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function uploadSource() {
    const { image, retry } = await client.img2img(model, {
      ...img2img,
      source: mustExist(img2img.source), // TODO: show an error if this doesn't exist
    }, upscale, highres);

    pushHistory(image, retry);
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

  const state = mustExist(useContext(StateContext));
  const model = useStore(state, (s) => s.img2imgModel);
  const source = useStore(state, (s) => s.img2img.source);
  const img2img = useStore(state, (s) => s.img2img);
  const highres = useStore(state, (s) => s.img2imgHighres);
  const upscale = useStore(state, (s) => s.img2imgUpscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHighres = useStore(state, (s) => s.setImg2ImgHighres);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setImg2ImgUpscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setModel = useStore(state, (s) => s.setImg2ImgModel);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <Profiles params={img2img} setParams={setImg2Img} highres={highres} setHighres={setHighres} upscale={upscale} setUpscale={setUpscale} />
      <ModelControl model={model} setModel={setModel} />
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
      <ImageControl selector={(s) => s.img2img} onChange={setImg2Img} />
      <Stack direction='row' spacing={2}>
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
          value={img2img.sourceFilter}
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
          value={img2img.strength}
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
          value={img2img.loopback}
          onChange={(value) => {
            setImg2Img({
              loopback: value,
            });
          }}
        />
      </Stack>
      <HighresControl highres={highres} setHighres={setHighres} />
      <UpscaleControl upscale={upscale} setUpscale={setUpscale} />
      <Button
        disabled={doesExist(source) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
