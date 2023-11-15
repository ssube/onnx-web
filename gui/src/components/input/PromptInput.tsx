import { Maybe, doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Chip, TextField } from '@mui/material';
import { Stack } from '@mui/system';
import { useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { STALE_TIME } from '../../config.js';
import { ClientContext, OnnxState, StateContext } from '../../state.js';
import { QueryMenu } from '../input/QueryMenu.js';
import { ModelResponse } from '../../types/api.js';

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

  const { t } = useTranslation();

  function addNetwork(type: string, name: string, weight = 1.0) {
    onChange({
      prompt: `<${type}:${name}:1.0> ${prompt}`,
      negativePrompt,
    });
  }

  function addToken(name: string) {
    onChange({
      prompt: `${prompt}, ${name}`,
    });
  }

  const networks = extractNetworks(prompt);
  const tokens = getNetworkTokens(models.data, networks);

  return <Stack spacing={2}>
    <TextField
      label={t('parameter.prompt')}
      variant='outlined'
      value={prompt}
      onChange={(event) => {
        props.onChange({
          prompt: event.target.value,
          negativePrompt,
        });
      }}
    />
    <Stack direction='row' spacing={2}>
      {tokens.map(([token, _weight]) => <Chip
        color={prompt.includes(token) ? 'primary' : 'default'}
        label={token}
        onClick={() => addToken(token)}
      />)}
    </Stack>
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
          addNetwork('inversion', name);
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
          addNetwork('lora', name);
        }}
      />
    </Stack>
  </Stack>;
}

export const ANY_TOKEN = /<([^>]+)>/g;

export type TokenList = Array<[string, number]>;

export interface PromptNetworks {
  inversion: TokenList;
  lora: TokenList;
}

export function extractNetworks(prompt: string): PromptNetworks {
  const inversion: TokenList = [];
  const lora: TokenList = [];

  for (const token of prompt.matchAll(ANY_TOKEN)) {
    const [_whole, match] = Array.from(token);
    const [type, name, weight, ..._rest] = match.split(':');

    switch (type) {
      case 'inversion':
        inversion.push([name, parseFloat(weight)]);
        break;
      case 'lora':
        lora.push([name, parseFloat(weight)]);
        break;
      default:
        // ignore others
    }
  }

  return {
    inversion,
    lora,
  };
}

// eslint-disable-next-line sonarjs/cognitive-complexity
export function getNetworkTokens(models: Maybe<ModelResponse>, networks: PromptNetworks): TokenList {
  const tokens: TokenList = [];

  if (doesExist(models)) {
    for (const [name, weight] of networks.inversion) {
      const model = models.networks.find((it) => it.type === 'inversion' && it.name === name);
      if (doesExist(model) && model.type === 'inversion') {
        tokens.push([model.token, weight]);
      }
    }

    for (const [name, weight] of networks.lora) {
      const model = models.networks.find((it) => it.type === 'lora' && it.name === name);
      if (doesExist(model) && model.type === 'lora') {
        for (const token of mustDefault(model.tokens, [])) {
          tokens.push([token, weight]);
        }
      }
    }
  }

  return tokens;
}
