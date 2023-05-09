import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { AnyFormat, ExtraSource } from '../../../types.js';

export interface ExtraSourceInputProps {
  key?: number | string;
  model: ExtraSource;

  onChange: (model: ExtraSource) => void;
  onRemove: (model: ExtraSource) => void;
}

export function ExtraSourceInput(props: ExtraSourceInputProps) {
  const { key, model, onChange, onRemove } = props;
  const { t } = useTranslation();

  return <Stack direction='row' spacing={2} key={key}>
    <TextField
      label={t('extras.name')}
      value={model.name}
      onChange={(event) => {
        onChange({
          ...model,
          name: event.target.value,
        });
      }}
    />
    <TextField
      label={t('extras.source')}
      value={model.source}
      onChange={(event) => {
        onChange({
          ...model,
          source: event.target.value,
        });
      }}
    />
    <Select
      label={t('extras.format')}
      value={model.format}
      onChange={(selection) => {
        onChange({
          ...model,
          format: selection.target.value as AnyFormat,
        });
      }}
    >
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
      <MenuItem value='json'>json</MenuItem>
      <MenuItem value='yaml'>yaml</MenuItem>
    </Select>
    <TextField
      label={t('extras.dest')}
      value={model.dest}
      onChange={(event) => {
        onChange({
          ...model,
          dest: event.target.value,
        });
      }}
    />
    <Button onClick={() => onRemove(model)}>{t('extras.remove')}</Button>
  </Stack>;
}
