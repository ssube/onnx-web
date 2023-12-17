import { Maybe, doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Chip, TextField } from '@mui/material';
import { Stack } from '@mui/system';
import { useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { STALE_TIME } from '../../config.js';
import { ClientContext, OnnxState, StateContext } from '../../state/full.js';
import { QueryMenu } from '../input/QueryMenu.js';
import { ModelResponse, NetworkModel } from '../../types/api.js';

const { useContext, useMemo } = React;

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

export interface PromptTextBlockProps extends PromptInputProps {
  models: Maybe<ModelResponse>;
}

export const PROMPT_GROUP = 75;

export function PromptTextBlock(props: PromptTextBlockProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { models, selector, onChange } = props;

  const { t } = useTranslation();
  const store = mustExist(useContext(StateContext));
  const { prompt, negativePrompt } = useStore(store, selector, shallow);

  function addToken(name: string) {
    onChange({
      prompt: `${prompt}, ${name}`,
    });
  }

  const tokens = useMemo(() => {
    const networks = extractNetworks(prompt);
    return getNetworkTokens(models, networks);
  }, [models, prompt]);

  return <Stack spacing={2}>
    <TextField
      label={t('parameter.prompt')}
      variant='outlined'
      value={prompt}
      onChange={(event) => {
        onChange({
          prompt: event.target.value,
          negativePrompt,
        });
      }}
    />
    <Stack direction='row' spacing={2}>
      {tokens.map((token) => <Chip
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
  </Stack>;
}

export function PromptInput(props: PromptInputProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { selector, onChange } = props;

  const store = mustExist(useContext(StateContext));
  const client = mustExist(useContext(ClientContext));
  const models = useQuery(['models'], async () => client.models(), {
    staleTime: STALE_TIME,
  });
  const wildcards = useQuery(['wildcards'], async () => client.wildcards(), {
    staleTime: STALE_TIME,
  });

  const { t } = useTranslation();

  function addNetwork(type: string, name: string, weight = 1.0) {
    const { prompt, negativePrompt } = selector(store.getState());
    onChange({
      negativePrompt,
      prompt: `<${type}:${name}:${weight.toFixed(2)}> ${prompt}`,
    });
  }

  function addWildcard(name: string) {
    const { prompt, negativePrompt } = selector(store.getState());
    onChange({
      negativePrompt,
      prompt: `${prompt}, __${name}__`,
    });
  }

  return <Stack spacing={2}>
    <PromptTextBlock
      models={models.data}
      onChange={onChange}
      selector={selector}
    />
    <Stack direction='row' spacing={2}>
      <QueryMenu
        id='inversion'
        labelKey='model.inversion'
        name={t('modelType.inversion')}
        query={{
          result: models,
          selector: (result) => filterNetworks(result.networks, 'inversion'),
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
          selector: (result) => filterNetworks(result.networks, 'lora'),
        }}
        onSelect={(name) => {
          addNetwork('lora', name);
        }}
      />
      <QueryMenu
        id='wildcard'
        labelKey='wildcard'
        name={t('wildcard')}
        query={{
          result: wildcards,
          selector: (result) => result,
        }}
        onSelect={(name) => {
          addWildcard(name);
        }}
      />
    </Stack>
  </Stack>;
}

export function filterNetworks(networks: Array<NetworkModel>, type: string): Array<string> {
  return networks.filter((network) => network.type === type).map((network) => network.name);
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
export function getNetworkTokens(models: Maybe<ModelResponse>, networks: PromptNetworks): Array<string> {
  const tokens: Set<string> = new Set();

  if (doesExist(models)) {
    for (const [name, _weight] of networks.inversion) {
      const model = models.networks.find((it) => it.type === 'inversion' && it.name === name);
      if (doesExist(model) && model.type === 'inversion') {
        tokens.add(model.token);
      }
    }

    for (const [name, _weight] of networks.lora) {
      const model = models.networks.find((it) => it.type === 'lora' && it.name === name);
      if (doesExist(model) && model.type === 'lora') {
        for (const token of mustDefault(model.tokens, [])) {
          tokens.add(token);
        }
      }
    }
  }

  return Array.from(tokens).sort();
}
