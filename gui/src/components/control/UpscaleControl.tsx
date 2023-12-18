import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { UpscaleParams } from '../../types/params.js';
import { NumericField } from '../input/NumericField.js';

export interface UpscaleControlProps {
  selectUpscale(state: OnnxState): UpscaleParams;
  setUpscale(params: Partial<UpscaleParams>): void;
}

export function UpscaleControl(props: UpscaleControlProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { selectUpscale, setUpscale } = props;

  const store = mustExist(useContext(StateContext));
  const upscale = useStore(store, selectUpscale);

  const { params } = mustExist(useContext(ConfigContext));
  const { t } = useTranslation();

  return <Stack direction='row' spacing={4}>
    <FormControlLabel
      label={t('parameter.upscale.label')}
      control={<Checkbox
        checked={upscale.enabled}
        value='check'
        onChange={(_event) => {
          setUpscale({
            enabled: upscale.enabled === false,
          });
        }}
      />}
    />
    <NumericField
      label={t('parameter.upscale.denoise')}
      decimal
      disabled={upscale.enabled === false}
      min={params.denoise.min}
      max={params.denoise.max}
      step={params.denoise.step}
      value={upscale.denoise}
      onChange={(denoise) => {
        setUpscale({
          denoise,
        });
      }}
    />
    <NumericField
      label={t('parameter.upscale.scale')}
      disabled={upscale.enabled === false}
      min={params.scale.min}
      max={params.scale.max}
      step={params.scale.step}
      value={upscale.scale}
      onChange={(scale) => {
        setUpscale({
          scale,
        });
      }}
    />
    <NumericField
      label={t('parameter.upscale.outscale')}
      disabled={upscale.enabled === false}
      min={params.outscale.min}
      max={params.outscale.max}
      step={params.outscale.step}
      value={upscale.outscale}
      onChange={(outscale) => {
        setUpscale({
          outscale,
        });
      }}
    />
    <FormControlLabel
      label={t('parameter.correction.label')}
      control={<Checkbox
        checked={upscale.faces}
        value='check'
        onChange={(event) => {
          setUpscale({
            faces: upscale.faces === false,
          });
        }}
      />}
    />
    <NumericField
      label={t('parameter.correction.strength')}
      decimal
      disabled={upscale.faces === false}
      min={params.faceStrength.min}
      max={params.faceStrength.max}
      step={params.faceStrength.step}
      value={upscale.faceStrength}
      onChange={(faceStrength) => {
        setUpscale({
          faceStrength,
        });
      }}
    />
    <NumericField
      label={t('parameter.correction.outscale')}
      disabled={upscale.faces === false}
      min={params.faceOutscale.min}
      max={params.faceOutscale.max}
      step={params.faceOutscale.step}
      value={upscale.faceOutscale}
      onChange={(faceOutscale) => {
        setUpscale({
          faceOutscale,
        });
      }}
    />
    <FormControl>
      <InputLabel id={'upscale-order'}>{t('parameter.upscale.order')}</InputLabel>
      <Select
        labelId={'upscale-order'}
        label={t('parameter.upscale.order')}
        value={upscale.upscaleOrder}
        onChange={(e) => {
          setUpscale({
            upscaleOrder: e.target.value,
          });
        }}
      >
        {Object.entries(params.upscaleOrder.keys).map(([key, name]) =>
          <MenuItem key={key} value={name}>{t(`upscaleOrder.${name}`)}</MenuItem>)
        }
      </Select>
    </FormControl>
  </Stack>;
}
