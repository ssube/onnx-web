import { mustExist } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import { Stack } from '@mui/system';
import { useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { STALE_TIME } from '../../config.js';
import { ClientContext, OnnxState, StateContext } from '../../state.js';
import { QueryMenu } from '../input/QueryMenu.js';

const { useContext } = React;

/**
 * @todo replace with a selector
 */
export interface PromptValue {
  prompt: string;
  negativePrompt?: string;
}

export interface PromptInputProps {
  selector(state: OnnxState): PromptValue;
  onChange(value: PromptValue): void;
}

export const PROMPT_GROUP = 75;

function splitPrompt(prompt: string): Array<string> {
  return prompt
    .split(',')
    .flatMap((phrase) => phrase.split(' '))
    .map((word) => word.trim())
    .filter((word) => word.length > 0);
}

export function PromptInput(props: PromptInputProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { selector, onChange } = props;

  const store = mustExist(useContext(StateContext));
  const { prompt, negativePrompt } = useStore(store, selector, shallow);

  const client = mustExist(useContext(ClientContext));
  const models = useQuery(['models'], async () => client.models(), {
    staleTime: STALE_TIME,
  });

  const tokens = splitPrompt(prompt);
  const groups = Math.ceil(tokens.length / PROMPT_GROUP);

  const { t } = useTranslation();
  const helper = t('input.prompt.tokens', {
    groups,
    tokens: tokens.length,
  });

  function addToken(type: string, name: string, weight = 1.0) {
    onChange({
      prompt: `<${type}:${name}:1.0> ${prompt}`,
      negativePrompt,
    });
  }

  return <Stack spacing={2}>
    <TextField
      label={t('parameter.prompt')}
      helperText={helper}
      variant='outlined'
      value={prompt}
      onChange={(event) => {
        props.onChange({
          prompt: event.target.value,
          negativePrompt,
        });
      }}
    />
    <TextField
      label={t('parameter.negativePrompt')}
      variant='outlined'
      value={negativePrompt}
      onChange={(event) => {
        props.onChange({
          prompt,
          negativePrompt: event.target.value,
        });
      }}
    />
    <Stack direction='row' spacing={2}>
      <QueryMenu
        id='inversion'
        labelKey='model.inversion'
        name={t('modelType.inversion')}
        query={{
          result: models,
          selector: (result) => result.networks.filter((network) => network.type === 'inversion').map((network) => network.name),
        }}
        onSelect={(name) => {
          addToken('inversion', name);
        }}
      />
      <QueryMenu
        id='lora'
        labelKey='model.lora'
        name={t('modelType.lora')}
        query={{
          result: models,
          selector: (result) => result.networks.filter((network) => network.type === 'lora').map((network) => network.name),
        }}
        onSelect={(name) => {
          addToken('lora', name);
        }}
      />
    </Stack>
  </Stack>;
}
