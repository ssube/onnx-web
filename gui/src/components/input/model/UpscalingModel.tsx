import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { UpscalingModel } from '../../../types.js';
import { NumericField } from '../NumericField.js';

export interface UpscalingModelInputProps {
  model: UpscalingModel;

  onChange: (model: UpscalingModel) => void;
}

export function UpscalingModelInput(props: UpscalingModelInputProps) {
  const { model, onChange } = props;

  return <Stack direction='row' spacing={2}>
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
        format: selection.target.value as 'ckpt',
      });
    }}>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <Select value={model.model} label='Type' onChange={(selection) => {
      onChange({
        ...model,
        model: selection.target.value as 'bsrgan',
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
  </Stack>;
}
