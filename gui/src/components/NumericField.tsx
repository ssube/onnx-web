import { doesExist } from '@apextoaster/js-utils';
import { Slider, Stack, TextField } from '@mui/material';
import * as React from 'react';

export function parseNumber(num: string, decimal = false): number {
  if (decimal) {
    return parseFloat(num);
  } else {
    return parseInt(num, 10);
  }
}

export interface ImageControlProps {
  decimal?: boolean;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;

  onChange?: (value: number) => void;
}

export function NumericField(props: ImageControlProps) {
  const { decimal, label, min, max, step, value } = props;
  return <Stack spacing={2}>
    <TextField
      label={label}
      variant='outlined'
      type='number'
      inputProps={{ min, max, step }}
      value={value}
      onChange={(event) => {
        if (doesExist(props.onChange)) {
          props.onChange(parseNumber(event.target.value, decimal));
        }
      }}
    />
    <Slider
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(_event, newValue) => {
        if (doesExist(props.onChange)) {
          if (Array.isArray(newValue)) {
            props.onChange(newValue[0]);
          } else {
            props.onChange(newValue);
          }
        }
      }}
    />
  </Stack>;
}
