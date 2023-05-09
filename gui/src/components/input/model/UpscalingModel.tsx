import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { ModelFormat, UpscalingArch, UpscalingModel } from '../../../types.js';
import { NumericField } from '../NumericField.js';

export interface UpscalingModelInputProps {
  key?: number | string;
  model: UpscalingModel;

  onChange: (model: UpscalingModel) => void;
  onRemove: (model: UpscalingModel) => void;
}

export function UpscalingModelInput(props: UpscalingModelInputProps) {
  const { key, model, onChange, onRemove } = props;

  return <Stack direction='row' spacing={2} key={key}>
    <TextField value={model.label} label='Label' onChange={(event) => {
      onChange({
        ...model,
        label: event.target.value,
      });
    }} />
    <TextField value={model.source} label='Source' onChange={(event) => {
      onChange({
        ...model,
        source: event.target.value,
      });
    }} />
    <Select value={model.format} label='Format' onChange={(selection) => {
      onChange({
        ...model,
        format: selection.target.value as ModelFormat,
      });
    }}>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <Select value={model.model} label='Type' onChange={(selection) => {
      onChange({
        ...model,
        model: selection.target.value as UpscalingArch,
      });
    }}>
      <MenuItem value='bsrgan'>BSRGAN</MenuItem>
      <MenuItem value='resrgan'>Real ESRGAN</MenuItem>
      <MenuItem value='swinir'>SwinIR</MenuItem>
    </Select>
    <NumericField
      label='Scale'
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
    <Button onClick={() => onRemove(model)}>Remove</Button>
  </Stack>;
}
