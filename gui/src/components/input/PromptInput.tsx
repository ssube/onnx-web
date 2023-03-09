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

export const PROMPT_LIMIT = 77;

export function PromptInput(props: PromptInputProps) {
  const { prompt = '', negativePrompt = '' } = props;
  const promptLength = prompt.split(' ').length;
  const error = promptLength > PROMPT_LIMIT;

  const { t } = useTranslation();

  function promptHelper() {
    const params = {
      current: promptLength,
      max: PROMPT_LIMIT,
    };

    if (error) {
      return t('input.prompt.error.length', params);
    } else {
      return t('input.prompt.tokens', params);
    }
  }

  return <Stack spacing={2}>
    <TextField
      error={error}
      label={t('parameter.prompt')}
      helperText={promptHelper()}
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
