import { doesExist, mustExist } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import { Stack } from '@mui/system';
import { useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { QueryMenu } from '../input/QueryMenu.js';
import { STALE_TIME } from '../../config.js';
import { ClientContext } from '../../state.js';

const { useContext } = React;

export interface PromptValue {
  prompt: string;
  negativePrompt?: string;
}

export interface PromptInputProps extends PromptValue {
  onChange: (value: PromptValue) => void;
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
  const { prompt = '', negativePrompt = '' } = props;

  const tokens = splitPrompt(prompt);
  const groups = Math.ceil(tokens.length / PROMPT_GROUP);

  const client = mustExist(useContext(ClientContext));
  const models = useQuery(['models'], async () => client.models(), {
    staleTime: STALE_TIME,
  });

  const { t } = useTranslation();
  const helper = t('input.prompt.tokens', {
    groups,
    tokens: tokens.length,
  });

  function addToken(type: string, name: string, weight = 1.0) {
    props.onChange({
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
        if (doesExist(props.onChange)) {
          props.onChange({
            prompt: event.target.value,
            negativePrompt,
          });
        }
      }}
    />
    <TextField
      label={t('parameter.negativePrompt')}
      variant='outlined'
      value={negativePrompt}
      onChange={(event) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            prompt,
            negativePrompt: event.target.value,
          });
        }
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
