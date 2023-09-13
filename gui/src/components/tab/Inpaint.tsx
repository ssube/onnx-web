import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Alert, Box, Button, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { IMAGE_FILTER, STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext, TabState } from '../../state.js';
import { BrushParams, HighresParams, InpaintParams, ModelParams, UpscaleParams } from '../../types/params.js';
import { Profiles } from '../Profiles.js';
import { HighresControl } from '../control/HighresControl.js';
import { ImageControl } from '../control/ImageControl.js';
import { ModelControl } from '../control/ModelControl.js';
import { OutpaintControl } from '../control/OutpaintControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';

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
    const state = store.getState();
    const { outpaint } = state;
    const inpaint = selectParams(state);

    if (outpaint.enabled) {
      const { image, retry } = await client.outpaint(model, {
        ...inpaint,
        ...outpaint,
        mask: mustExist(mask),
        source: mustExist(source),
      }, selectUpscale(state), selectHighres(state));

      pushHistory(image, retry);
    } else {
      const { image, retry } = await client.inpaint(model, {
        ...inpaint,
        mask: mustExist(mask),
        source: mustExist(source),
      }, selectUpscale(state), selectHighres(state));

      pushHistory(image, retry);
    }
  }

  function preventInpaint(): boolean {
    return doesExist(source) === false || doesExist(mask) === false;
  }

  function supportsInpaint(): boolean {
    return model.model.includes('inpaint');
  }

  const store = mustExist(useContext(StateContext));
  const { pushHistory, setBrush, setHighres, setModel, setInpaint, setUpscale } = useStore(store, selectActions, shallow);
  const { source, mask, strength, noise, filter, tileOrder, fillColor } = useStore(store, selectReactParams, shallow);
  const model = useStore(store, selectModel);
  const brush = useStore(store, selectBrush);

  const { t } = useTranslation();

  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries(['ready']),
  });

  function renderBanner() {
    if (supportsInpaint()) {
      return undefined;
    } else {
      return <Alert severity='warning'>{t('error.inpaint.support')}</Alert>;
    }
  }

  return <Box>
    <Stack spacing={2}>
      <Profiles
        selectHighres={selectHighres}
        selectParams={selectParams}
        selectUpscale={selectUpscale}
        setParams={setInpaint}
        setHighres={setHighres}
        setUpscale={setUpscale}
      />
      <ModelControl model={model} setModel={setModel} />
      {renderBanner()}
      <ImageInput
        filter={IMAGE_FILTER}
        image={source}
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
        image={mask}
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
        source={source}
        mask={mask}
        onSave={(file) => {
          setInpaint({
            mask: file,
          });
        }}
        setBrush={setBrush}
      />
      <ImageControl
        selector={selectParams}
        onChange={(newParams) => {
          setInpaint(newParams);
        }}
      />
      <NumericField
        label={t('parameter.strength')}
        min={params.strength.min}
        max={params.strength.max}
        step={params.strength.step}
        value={strength}
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
          value={filter}
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
          value={noise}
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
            value={tileOrder}
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
                defaultValue={fillColor}
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
      <HighresControl selectHighres={selectHighres} setHighres={setHighres} />
      <UpscaleControl selectUpscale={selectUpscale} setUpscale={setUpscale} />
      <Button
        disabled={preventInpaint()}
        variant='contained'
        onClick={() => upload.mutate()}
        color={supportsInpaint() ? undefined : 'warning'}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    pushHistory: state.pushHistory,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setBrush: state.setInpaintBrush,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setHighres: state.setInpaintHighres,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setModel: state.setInpaintModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setInpaint: state.setInpaint,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setInpaintUpscale,
  };
}

export function selectBrush(state: OnnxState): BrushParams {
  return state.inpaintBrush;
}

export function selectModel(state: OnnxState): ModelParams {
  return state.inpaintModel;
}

export function selectParams(state: OnnxState): TabState<InpaintParams> {
  return state.inpaint;
}

export function selectReactParams(state: OnnxState) {
  return {
    source: state.inpaint.source,
    mask: state.inpaint.mask,
    strength: state.inpaint.strength,
    noise: state.inpaint.noise,
    filter: state.inpaint.filter,
    tileOrder: state.inpaint.tileOrder,
    fillColor: state.inpaint.fillColor,
  };
}

export function selectHighres(state: OnnxState): HighresParams {
  return state.inpaintHighres;
}

export function selectUpscale(state: OnnxState): UpscaleParams {
  return state.inpaintUpscale;
}
