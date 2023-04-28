import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function HighresControl() {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const highres = useStore(state, (s) => s.highres);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHighres = useStore(state, (s) => s.setHighres);
  const { t } = useTranslation();

  return <Stack direction='row' spacing={4}>
    <FormControlLabel
      label={t('parameter.tiledVAE')}
      control={<Checkbox
        checked={highres.tiledVAE}
        value='check'
        onChange={(event) => {
          setHighres({
            tiledVAE: highres.tiledVAE === false,
          });
        }}
      />}
    />
    <FormControlLabel
      label={t('parameter.highres.label')}
      control={<Checkbox
        checked={highres.enabled}
        value='check'
        onChange={(event) => {
          setHighres({
            enabled: highres.enabled === false,
          });
        }}
      />}
    />
    <NumericField
      label={t('parameter.highres.steps')}
      disabled={highres.enabled === false}
      min={params.highresSteps.min}
      max={params.highresSteps.max}
      step={params.highresSteps.step}
      value={highres.highresSteps}
      onChange={(steps) => {
        setHighres({
          highresSteps: steps,
        });
      }}
    />
    <NumericField
      label={t('parameter.highres.scale')}
      disabled={highres.enabled === false}
      min={params.highresScale.min}
      max={params.highresScale.max}
      step={params.highresScale.step}
      value={highres.highresScale}
      onChange={(scale) => {
        setHighres({
          highresScale: scale,
        });
      }}
    />
    <NumericField
      label={t('parameter.highres.strength')}
      decimal
      disabled={highres.enabled === false}
      min={params.highresStrength.min}
      max={params.highresStrength.max}
      step={params.highresStrength.step}
      value={highres.highresStrength}
      onChange={(strength) => {
        setHighres({
          highresStrength: strength,
        });
      }}
    />
    <FormControl>
      <InputLabel id={'highres-method'}>{t('parameter.highres.method')}</InputLabel>
      <Select
        disabled={highres.enabled === false}
        labelId={'highres-method'}
        label={t('parameter.highres.method')}
        value={highres.highresMethod}
        onChange={(e) => {
          setHighres({
            highresMethod: e.target.value,
          });
        }}
      >
        {Object.entries(params.highresMethod.keys).map(([key, name]) =>
          <MenuItem key={key} value={name}>{t(`highresMethod.${name}`)}</MenuItem>)
        }
      </Select>
    </FormControl>
    <NumericField
      label={t('parameter.highres.iterations')}
      disabled={highres.enabled === false}
      min={params.highresIterations.min}
      max={params.highresIterations.max}
      step={params.highresIterations.step}
      value={highres.highresIterations}
      onChange={(iterations) => {
        setHighres({
          highresIterations: iterations,
        });
      }}
    />
  </Stack>;
}
