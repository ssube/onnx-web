import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useQuery } from 'react-query';
import { useStore } from 'zustand';

import { BaseImgParams } from '../../client.js';
import { STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state.js';
import { SCHEDULER_LABELS } from '../../strings.js';
import { NumericField } from '../input/NumericField.js';
import { QueryList } from '../input/QueryList.js';

const { useContext } = React;

export const PROMPT_LIMIT = 70;

export interface ImageControlProps {
  selector: (state: OnnxState) => BaseImgParams;

  onChange?: (params: BaseImgParams) => void;
}

/**
 * doesn't need to use state, the parent component knows which params to pass
 */
export function ImageControl(props: ImageControlProps) {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const controlState = useStore(state, props.selector);

  const client = mustExist(useContext(ClientContext));
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const promptLength = controlState.prompt.split(' ').length;
  const error = promptLength > PROMPT_LIMIT;

  function promptHelper() {
    if (error) {
      return `Too many tokens: ${promptLength}/${PROMPT_LIMIT}`;
    } else {
      return `Tokens: ${promptLength}/${PROMPT_LIMIT}`;
    }
  }

  return <Stack spacing={2}>
    <QueryList
      id='schedulers'
      labels={SCHEDULER_LABELS}
      name='Scheduler'
      query={{
        result: schedulers,
      }}
      value={mustDefault(controlState.scheduler, '')}
      onChange={(value) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            ...controlState,
            scheduler: value,
          });
        }
      }}
    />
    <Stack direction='row' spacing={4}>
      <NumericField
        decimal
        label='CFG'
        min={params.cfg.min}
        max={params.cfg.max}
        step={params.cfg.step}
        value={controlState.cfg}
        onChange={(cfg) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              cfg,
            });
          }
        }}
      />
      <NumericField
        label='Steps'
        min={params.steps.min}
        max={params.steps.max}
        step={params.steps.step}
        value={controlState.steps}
        onChange={(steps) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              steps,
            });
          }
        }}
      />
      <NumericField
        label='Seed'
        min={params.seed.min}
        max={params.seed.max}
        step={params.seed.step}
        value={controlState.seed}
        onChange={(seed) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              seed,
            });
          }
        }}
      />
      <Button
        variant='outlined'
        startIcon={<Casino />}
        onClick={() => {
          const seed = Math.floor(Math.random() * params.seed.max);
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              seed,
            });
          }
        }}
      >
        New Seed
      </Button>
    </Stack>
    <TextField
      error={error}
      label='Prompt'
      helperText={promptHelper()}
      variant='outlined'
      value={controlState.prompt}
      onChange={(event) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            ...controlState,
            prompt: event.target.value,
          });
        }
      }}
    />
    <TextField
      label='Negative Prompt'
      variant='outlined'
      value={controlState.negativePrompt}
      onChange={(event) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            ...controlState,
            negativePrompt: event.target.value,
          });
        }
      }}
    />
  </Stack>;
}
