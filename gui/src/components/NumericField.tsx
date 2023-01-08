import { doesExist } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import * as React from 'react';

export function parseNumber(num: string, decimal=false): number {
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
  step: number | 'any';
  value: number;

  onChange?: (value: number) => void;
}

export function NumericField(props: ImageControlProps) {
  const { label, min, max, step, value } = props;
  return <TextField
    label={label}
    variant='outlined'
    type='number'
    inputProps={{ min, max, step }}
    value={value}
    onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange(parseNumber(event.target.value, props.decimal));
      }
    }}
  />;
}
