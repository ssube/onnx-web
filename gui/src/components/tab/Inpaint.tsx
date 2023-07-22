import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Alert, Box, Button, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { IMAGE_FILTER, STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext, TabState } from '../../state.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { OutpaintControl } from '../control/OutpaintControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';
import { Profiles } from '../Profiles.js';
import { ModelParams, InpaintParams, HighresParams, UpscaleParams } from '../../client/types.js';

export function Inpaint() {
  const { params } = mustExist(useContext(ConfigContext));
  const client = mustExist(useContext(ClientContext));

  const filters = useQuery(['filters'], async () => client.filters(), {
    staleTime: STALE_TIME,
  });
  const noises = useQuery(['noises'], async () => client.noises(), {
    staleTime: STALE_TIME,
  });

  async function uploadSource(): Promise<void> {
    if (outpaint.enabled) {
      const { image, retry } = await client.outpaint(model, {
        ...inpaint,
        ...outpaint,
        mask: mustExist(inpaint.mask),
        source: mustExist(inpaint.source),
      }, upscale, highres);

      pushHistory(image, retry);
    } else {
      const { image, retry } = await client.inpaint(model, {
        ...inpaint,
        mask: mustExist(inpaint.mask),
        source: mustExist(inpaint.source),
      }, upscale, highres);

      pushHistory(image, retry);
    }
  }

  function preventInpaint(): boolean {
    return doesExist(inpaint.source) === false || doesExist(inpaint.mask) === false;
  }

  function supportsInpaint(): boolean {
    return model.model.includes('inpaint');
  }

  const state = mustExist(useContext(StateContext));
  const inpaint = useStore(state, selectParams);
  const highres = useStore(state, selectHighres);
  const model = useStore(state, selectModel);
  const upscale = useStore(state, selectUpscale);
  const outpaint = useStore(state, (s) => s.outpaint);
  const brush = useStore(state, (s) => s.inpaintBrush);

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBrush = useStore(state, (s) => s.setInpaintBrush);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setModel = useStore(state, (s) => s.setInpaintModel);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHighres = useStore(state, (s) => s.setInpaintHighres);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setInpaintUpscale);

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);
  const { t } = useTranslation();

  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries([ 'ready' ]),
  });

  function renderBanner() {
    if (supportsInpaint()) {
      return undefined;
    } else {
      return <Alert severity="warning">{t('error.inpaint.support')}</Alert>;
    }
  }

  return <Box>
    <Stack spacing={2}>
      <Profiles
        params={inpaint}
        setParams={setInpaint}
        highres={highres}
        setHighres={setHighres}
        upscale={upscale}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} />
      {renderBanner()}
      <ImageInput
        filter={IMAGE_FILTER}
        image={inpaint.source}
        label={t('input.image.source')}
        hideSelection={true}
        onChange={(file) => {
          setInpaint({
            source: file,
          });
        }}
      />
      <ImageInput
        filter={IMAGE_FILTER}
        image={inpaint.mask}
        label={t('input.image.mask')}
        hideSelection={true}
        onChange={(file) => {
          setInpaint({
            mask: file,
          });
        }}
      />
      <MaskCanvas
        brush={brush}
        source={inpaint.source}
        mask={inpaint.mask}
        onSave={(file) => {
          setInpaint({
            mask: file,
          });
        }}
        setBrush={setBrush}
      />
      <ImageControl
        selector={(s) => s.inpaint}
        onChange={(newParams) => {
          setInpaint(newParams);
        }}
      />
      <NumericField
        label={t('parameter.strength')}
        min={params.strength.min}
        max={params.strength.max}
        step={params.strength.step}
        value={inpaint.strength}
        onChange={(value) => {
          setInpaint({
            strength: value,
          });
        }}
      />
      <Stack direction='row' spacing={2}>
        <QueryList
          id='masks'
          labelKey={'maskFilter'}
          name={t('parameter.maskFilter')}
          query={{
            result: filters,
            selector: (f) => f.mask,
          }}
          value={inpaint.filter}
          onChange={(newFilter) => {
            setInpaint({
              filter: newFilter,
            });
          }}
        />
        <QueryList
          id='noises'
          labelKey={'noiseSource'}
          name={t('parameter.noiseSource')}
          query={{
            result: noises,
          }}
          value={inpaint.noise}
          onChange={(newNoise) => {
            setInpaint({
              noise: newNoise,
            });
          }}
        />
        <FormControl>
          <InputLabel id={'outpaint-tiling'}>Tile Order</InputLabel>
          <Select
            labelId={'outpaint-tiling'}
            label={t('parameter.tileOrder')}
            value={inpaint.tileOrder}
            onChange={(e) => {
              setInpaint({
                tileOrder: e.target.value,
              });
            }}
          >
            {Object.entries(params.tileOrder.keys).map(([_key, name]) =>
              <MenuItem key={name} value={name}>{t(`tileOrder.${name}`)}</MenuItem>)
            }
          </Select>
        </FormControl>
        <Stack direction='row' spacing={2}>
          <FormControlLabel
            label={t('parameter.fillColor')}
            sx={{ mx: 1 }}
            control={
              <input
                defaultValue={inpaint.fillColor}
                name='fill-color'
                type='color'
                onBlur={(event) => {
                  setInpaint({
                    fillColor: event.target.value,
                  });
                }}
              />
            }
          />
        </Stack>
      </Stack>
      <OutpaintControl />
      <HighresControl highres={highres} setHighres={setHighres} />
      <UpscaleControl upscale={upscale} setUpscale={setUpscale} />
      <Button
        disabled={preventInpaint()}
        variant='contained'
        onClick={() => upload.mutate()}
        color={supportsInpaint() ? undefined : 'warning'}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}

export function selectModel(state: OnnxState): ModelParams {
  return state.inpaintModel;
}

export function selectParams(state: OnnxState): TabState<InpaintParams> {
  return state.inpaint;
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.inpaintHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.inpaintUpscale;
}
