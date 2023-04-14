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

export function Img2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function uploadSource() {
    const { model, img2img, upscale } = state.getState();
    const { image, retry } = await client.img2img(model, {
      ...img2img,
      source: mustExist(img2img.source), // TODO: show an error if this doesn't exist
    }, upscale);

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


  const state = mustExist(useContext(StateContext));
  const source = useStore(state, (s) => s.img2img.source);
  const sourceFilter = useStore(state, (s) => s.img2img.sourceFilter);
  const strength = useStore(state, (s) => s.img2img.strength);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} image={source} label={t('input.image.source')} onChange={(file) => {
        setImg2Img({
          source: file,
        });
      }} />
      <ImageControl selector={(s) => s.img2img} onChange={setImg2Img} />
      <Stack direction='row' spacing={2}>
        <QueryList
          id='sources'
          labelKey={'sourceFilter'}
          name={t('parameter.sourceFilter')}
          query={{
            result: filters,
            selector: (f) => f.source,
          }}
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
      </Stack>
      <UpscaleControl />
      <Button
        disabled={doesExist(source) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
