import { doesExist, Maybe } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import { Stack } from '@mui/system';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

export interface PromptValue {
  prompt: string;
  negativePrompt?: string;
}

export interface PromptInputProps extends PromptValue {
  onChange?: Maybe<(value: PromptValue) => void>;
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

  const { t } = useTranslation();
  const helper = t('input.prompt.tokens', {
    groups,
    tokens: tokens.length,
  });

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
  </Stack>;
}
