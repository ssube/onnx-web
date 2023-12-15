import { mustDefault, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { IMAGE_FILTER } from '../../config.js';
import { BLEND_SOURCES } from '../../constants.js';
import { ClientContext, OnnxState, StateContext } from '../../state/full.js';
import { TabState } from '../../state/types.js';
import { BlendParams, BrushParams, ModelParams, UpscaleParams } from '../../types/params.js';
import { range } from '../../utils.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';

export function Blend() {
  async function uploadSource() {
    const { blend, blendModel, blendUpscale } = store.getState();
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

  const store = mustExist(useContext(StateContext));
  const { pushHistory, setBlend, setBrush, setUpscale } = useStore(store, selectActions, shallow);
  const brush = useStore(store, selectBrush);
  const blend = useStore(store, selectParams);
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
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <Button
        disabled={sources.length < 2}
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
    setBlend: state.setBlend,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setBrush: state.setBlendBrush,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setBlendUpscale,
  };
}

export function selectBrush(state: OnnxState): BrushParams {
  return state.blendBrush;
}

export function selectModel(state: OnnxState): ModelParams {
  return state.blendModel;
}

export function selectParams(state: OnnxState): TabState<BlendParams> {
  return state.blend;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.blendUpscale;
}
