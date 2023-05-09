import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { DiffusionModel, ModelFormat } from '../../../types.js';

export interface DiffusionModelInputProps {
  key?: number | string;
  model: DiffusionModel;

  onChange: (model: DiffusionModel) => void;
  onRemove: (model: DiffusionModel) => void;
}

export function DiffusionModelInput(props: DiffusionModelInputProps) {
  const { key, model, onChange, onRemove } = props;
  const { t } = useTranslation();

  return <Stack direction='row' spacing={2} key={key}>
    <TextField
      label={t('extras.label')}
      value={model.label}
      onChange={(event) => {
        onChange({
          ...model,
          label: event.target.value,
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
          format: selection.target.value as ModelFormat,
        });
      }}
    >
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <Button onClick={() => onRemove(model)}>{t('extras.remove')}</Button>
  </Stack>;
}
