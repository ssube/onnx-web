import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { ExtraNetwork, ModelFormat, NetworkModel, NetworkType } from '../../../types.js';

export interface ExtraNetworkInputProps {
  key?: number | string;
  model: ExtraNetwork;

  onChange: (model: ExtraNetwork) => void;
  onRemove: (model: ExtraNetwork) => void;
}

export function ExtraNetworkInput(props: ExtraNetworkInputProps) {
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
      <MenuItem value='bin'>bin</MenuItem>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <Select
      label={t('extras.type')}
      value={model.type}
      onChange={(selection) => {
        onChange({
          ...model,
          type: selection.target.value as NetworkType,
        });
      }}
    >
      <MenuItem value='inversion'>{t('modelType.inversion')}</MenuItem>
      <MenuItem value='lora'>{t('modelType.lora')}</MenuItem>
    </Select>
    <Select
      label={t('extras.model')}
      value={model.model}
      onChange={(selection) => {
        onChange({
          ...model,
          model: selection.target.value as NetworkModel,
        });
      }}
    >
      <MenuItem value='sd-scripts'>sd-scripts</MenuItem>
      <MenuItem value='concept'>concept</MenuItem>
      <MenuItem value='embeddings'>embeddings</MenuItem>
    </Select>
    <Button onClick={() => onRemove(model)}>{t('extras.remove')}</Button>
  </Stack>;
}
