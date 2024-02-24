/* eslint-disable camelcase */
import { mustDefault, mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack, TextField } from '@mui/material';
import { useTranslation } from 'react-i18next';
import * as React from 'react';
import { useContext } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { STALE_TIME, STANDARD_SPACING } from '../../constants.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state/full.js';
import { ExperimentalParams } from '../../types/params.js';

export interface ExperimentalControlProps {
  selectExperimental(state: OnnxState): ExperimentalParams;
  setExperimental(params: Record<string, unknown>): void;
}

export function ExperimentalControl(props: ExperimentalControlProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { selectExperimental, setExperimental } = props;

  const store = mustExist(React.useContext(StateContext));
  const experimental = useStore(store, selectExperimental);

  const { params } = mustExist(useContext(ConfigContext));
  const { t } = useTranslation();

  const client = mustExist(React.useContext(ClientContext));
  const filters = useQuery(['filters'], async () => client.filters(), {
    staleTime: STALE_TIME,
  });

  return <Stack spacing={STANDARD_SPACING}>
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <FormControlLabel
        label={t('experimental.prompt_editing.label')}
        control={
          <Checkbox
            checked={experimental.promptEditing.enabled}
            value='check'
            onChange={(event) => {
              setExperimental({
                promptEditing: {
                  ...experimental.promptEditing,
                  enabled: experimental.promptEditing.enabled === false,
                },
              });
            }}
          />}
      />
      <QueryList
        disabled={experimental.promptEditing.enabled === false}
        id='prompt_filters'
        labelKey='model.prompt'
        name={t('experimental.prompt_editing.filter')}
        query={{
          result: filters,
          selector: (f) => f.prompt,
        }}
        value={mustDefault(experimental.promptEditing.filter, '')}
        onChange={(prompt_filter) => {
          setExperimental({
            promptEditing: {
              ...experimental.promptEditing,
              filter: prompt_filter,
            },
          });
        }}
      />
      <TextField
        disabled={experimental.promptEditing.enabled === false}
        label={t('experimental.prompt_editing.remove_tokens')}
        variant='outlined'
        value={experimental.promptEditing.removeTokens}
        onChange={(event) => {
          setExperimental({
            promptEditing: {
              ...experimental.promptEditing,
              removeTokens: event.target.value,
            },
          });
        }}
      />
      <TextField
        disabled={experimental.promptEditing.enabled === false}
        label={t('experimental.prompt_editing.add_suffix')}
        variant='outlined'
        value={experimental.promptEditing.addSuffix}
        onChange={(event) => {
          setExperimental({
            promptEditing: {
              ...experimental.promptEditing,
              addSuffix: event.target.value,
            },
          });
        }}
      />
      <NumericField
        disabled={experimental.promptEditing.enabled === false}
        label={t('experimental.prompt_editing.min_length')}
        min={params.promptEditing.minLength.min}
        max={params.promptEditing.minLength.max}
        step={params.promptEditing.minLength.step}
        value={experimental.promptEditing.minLength}
        onChange={(prompt_editing_min_length) => {
          setExperimental({
            promptEditing: {
              ...experimental.promptEditing,
              minLength: prompt_editing_min_length,
            },
          });
        }}
      />
    </Stack>
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <FormControlLabel
        label={t('experimental.latent_symmetry.label')}
        control={
          <Checkbox
            checked={experimental.latentSymmetry.enabled}
            value='check'
            onChange={(event) => {
              setExperimental({
                latentSymmetry: {
                  ...experimental.latentSymmetry,
                  enabled: experimental.latentSymmetry.enabled === false,
                },
              });
            }}
          />}
      />
      <NumericField
        decimal
        disabled={experimental.latentSymmetry.enabled === false}
        label={t('experimental.latent_symmetry.gradient_start')}
        min={params.latentSymmetry.gradientStart.min}
        max={params.latentSymmetry.gradientStart.max}
        step={params.latentSymmetry.gradientStart.step}
        value={experimental.latentSymmetry.gradientStart}
        onChange={(latent_symmetry_gradient_start) => {
          setExperimental({
            latentSymmetry: {
              ...experimental.latentSymmetry,
              gradientStart: latent_symmetry_gradient_start,
            },
          });
        }}
      />
      <NumericField
        decimal
        disabled={experimental.latentSymmetry.enabled === false}
        label={t('experimental.latent_symmetry.gradient_end')}
        min={params.latentSymmetry.gradientEnd.min}
        max={params.latentSymmetry.gradientEnd.max}
        step={params.latentSymmetry.gradientEnd.step}
        value={experimental.latentSymmetry.gradientEnd}
        onChange={(latent_symmetry_gradient_end) => {
          setExperimental({
            latentSymmetry: {
              ...experimental.latentSymmetry,
              gradientEnd: latent_symmetry_gradient_end,
            },
          });
        }}
      />
      <NumericField
        decimal
        disabled={experimental.latentSymmetry.enabled === false}
        label={t('experimental.latent_symmetry.line_of_symmetry')}
        min={params.latentSymmetry.lineOfSymmetry.min}
        max={params.latentSymmetry.lineOfSymmetry.max}
        step={params.latentSymmetry.lineOfSymmetry.step}
        value={experimental.latentSymmetry.lineOfSymmetry}
        onChange={(latent_symmetry_line_of_symmetry) => {
          setExperimental({
            latentSymmetry: {
              ...experimental.latentSymmetry,
              lineOfSymmetry: latent_symmetry_line_of_symmetry,
            },
          });
        }}
      />
    </Stack>
  </Stack>;
}
