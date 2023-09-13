import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { ModelFormat, UpscalingArch, UpscalingModel } from '../../../types/model.js';
import { NumericField } from '../NumericField.js';

export interface UpscalingModelInputProps {
  key?: number | string;
  model: UpscalingModel;

  onChange: (model: UpscalingModel) => void;
  onRemove: (model: UpscalingModel) => void;
}

export function UpscalingModelInput(props: UpscalingModelInputProps) {
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
    <Select
      label={t('extras.model')}
      value={model.model}
      onChange={(selection) => {
        onChange({
          ...model,
          model: selection.target.value as UpscalingArch,
        });
      }}
    >
      <MenuItem value='bsrgan'>BSRGAN</MenuItem>
      <MenuItem value='resrgan'>Real ESRGAN</MenuItem>
      <MenuItem value='swinir'>SwinIR</MenuItem>
    </Select>
    <NumericField
      label={t('extras.scale')}
      min={1}
      max={4}
      step={1}
      value={model.scale}
      onChange={(value) => {
        onChange({
          ...model,
          scale: value,
        });
      }}
    />
    <Button onClick={() => onRemove(model)}>{t('extras.remove')}</Button>
  </Stack>;
}
