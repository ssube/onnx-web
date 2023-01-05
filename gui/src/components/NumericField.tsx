import { doesExist } from '@apextoaster/js-utils';
import { TextField } from '@mui/material';
import * as React from 'react';

export interface ImageControlProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;

  onChange?: (value: number) => void;
}

export function NumericField(props: ImageControlProps) {
  const { label, min, max, step, value } = props;
  return <TextField
    label={label}
    variant="outlined"
    type="number"
    inputProps={{ min, max, step }}
    value={value}
    onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange(parseInt(event.target.value, 10));
      }
    }}
  />;
}
