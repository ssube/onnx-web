import { doesExist, Maybe } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import { Stack } from '@mui/system';
import * as React from 'react';

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

  function promptHelper() {
    if (error) {
      return `Too many tokens: ${promptLength}/${PROMPT_LIMIT}`;
    } else {
      return `Tokens: ${promptLength}/${PROMPT_LIMIT}`;
    }
  }

  return <Stack spacing={2}>
    <TextField
      error={error}
      label='Prompt'
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
      label='Negative Prompt'
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
