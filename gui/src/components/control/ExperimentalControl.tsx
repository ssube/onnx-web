/* eslint-disable camelcase */
import { mustDefault, mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack, TextField } from '@mui/material';
import { useTranslation } from 'react-i18next';
import * as React from 'react';
import { useQuery } from '@tanstack/react-query';

import { STALE_TIME, STANDARD_SPACING } from '../../constants.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';
import { ClientContext } from '../../state/full.js';

export interface ExperimentalControlProps {
  setExperimental(params: Record<string, unknown>): void;
}

export function ExperimentalControl(props: ExperimentalControlProps) {
  const { t } = useTranslation();
  const state = {
    latent_symmetry: false,
    latent_symmetry_gradient_start: 0.1,
    latent_symmetry_gradient_end: 0.3,
    latent_symmetry_line_of_symmetry: 0.5,
    prompt_editing: false,
    prompt_filter: '',
    remove_tokens: '',
    add_suffix: '',
  };

  const client = mustExist(React.useContext(ClientContext));
  const filters = useQuery(['filters'], async () => client.filters(), {
    staleTime: STALE_TIME,
  });

  return <Stack spacing={STANDARD_SPACING}>
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <FormControlLabel
        label={t('experimental.latent_symmetry.label')}
        control={
          <Checkbox
            checked={state.latent_symmetry}
            value='check'
            onChange={(event) => {
              props.setExperimental({
                latent_symmetry: state.latent_symmetry === false,
              });
            }}
          />}
      />
      <NumericField
        decimal
        label={t('experimental.latent_symmetry.gradient_start')}
        min={0}
        max={0.5}
        step={0.01}
        value={state.latent_symmetry_gradient_start}
        onChange={(latent_symmetry_gradient_start) => {
          props.setExperimental({
            latent_symmetry_gradient_start,
          });
        }}
      />
      <NumericField
        decimal
        label={t('experimental.latent_symmetry.gradient_end')}
        min={0}
        max={0.5}
        step={0.01}
        value={state.latent_symmetry_gradient_end}
        onChange={(latent_symmetry_gradient_end) => {
          props.setExperimental({
            latent_symmetry_gradient_end,
          });
        }}
      />
      <NumericField
        decimal
        label={t('experimental.latent_symmetry.line_of_symmetry')}
        min={0}
        max={1}
        step={0.01}
        value={state.latent_symmetry_line_of_symmetry}
        onChange={(latent_symmetry_line_of_symmetry) => {
          props.setExperimental({
            latent_symmetry_line_of_symmetry,
          });
        }}
      />
    </Stack>
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <FormControlLabel
        label={t('experimental.prompt_editing.label')}
        control={
          <Checkbox
            checked={state.prompt_editing}
            value='check'
            onChange={(event) => {
              props.setExperimental({
                prompt_editing: state.prompt_editing === false,
              });
            }}
          />}
      />
      <QueryList
        id='prompt_filters'
        labelKey='prompt_filter'
        name={t('experimental.prompt_editing.prompt_filters')}
        query={{
          result: filters,
          selector: (f) => f.prompt,
        }}
        value={mustDefault(state.prompt_filter, '')}
        onChange={(prompt_filter) => {
          props.setExperimental({
            prompt_filter,
          });
        }}
      />
      <TextField
        label={t('experimental.prompt_editing.remove_tokens')}
        variant='outlined'
        value={state.remove_tokens}
        onChange={(event) => {
          props.setExperimental({
            remove_tokens: event.target.value,
          });
        }}
      />
      <TextField
        label={t('experimental.prompt_editing.add_suffix')}
        variant='outlined'
        value={state.add_suffix}
        onChange={(event) => {
          props.setExperimental({
            add_suffix: event.target.value,
          });
        }}
      />
    </Stack>
  </Stack>;
}
