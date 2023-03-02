import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER, STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { ImageControl } from '../control/ImageControl.js';
import { OutpaintControl } from '../control/OutpaintControl.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';

export function Inpaint() {
  const { params } = mustExist(useContext(ConfigContext));
  const client = mustExist(useContext(ClientContext));

  const masks = useQuery('masks', async () => client.masks(), {
    staleTime: STALE_TIME,
  });
  const noises = useQuery('noises', async () => client.noises(), {
    staleTime: STALE_TIME,
  });

  async function uploadSource(): Promise<void> {
    // these are not watched by the component, only sent by the mutation
    const { model, inpaint, outpaint, upscale } = state.getState();

    if (outpaint.enabled) {
      const output = await client.outpaint(model, {
        ...inpaint,
        ...outpaint,
        mask: mustExist(mask),
        source: mustExist(source),
      }, upscale);

      setLoading(output);
    } else {
      const output = await client.inpaint(model, {
        ...inpaint,
        mask: mustExist(mask),
        source: mustExist(source),
      }, upscale);

      setLoading(output);
    }
  }

  const state = mustExist(useContext(StateContext));
  const fillColor = useStore(state, (s) => s.inpaint.fillColor);
  const filter = useStore(state, (s) => s.inpaint.filter);
  const noise = useStore(state, (s) => s.inpaint.noise);
  const mask = useStore(state, (s) => s.inpaint.mask);
  const source = useStore(state, (s) => s.inpaint.source);
  const strength = useStore(state, (s) => s.inpaint.strength);
  const tileOrder = useStore(state, (s) => s.inpaint.tileOrder);

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.pushLoading);
  const { t } = useTranslation();

  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  return <Box>
    <Stack spacing={2}>
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
        source={source}
        mask={mask}
        onSave={(file) => {
          setInpaint({
            mask: file,
          });
        }}
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
          showEmpty={true}
          name={t('parameter.maskFilter')}
          query={{
            result: masks,
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
            {Object.entries(params.tileOrder.keys).map(([key, name]) =>
              <MenuItem key={key} value={key}>{t(`tileOrder.${name}`)}</MenuItem>)
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
      <UpscaleControl />
      <Button
        disabled={doesExist(source) === false || doesExist(mask) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >{t('generate')}</Button>
    </Stack>
  </Box>;
}
