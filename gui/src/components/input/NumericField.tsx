import { doesExist } from '@apextoaster/js-utils';
import { Slider, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

export function parseNumber(num: string, decimal = false): number {
  if (decimal) {
    return parseFloat(num);
  } else {
    return parseInt(num, 10);
  }
}

export interface ImageControlProps {
  decimal?: boolean;
  disabled?: boolean;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;

  onChange?: (value: number) => void;
}

export function NumericField(props: ImageControlProps) {
  const { decimal = false, disabled = false, label, min, max, step, value } = props;
  const error = (value < min) || (value > max);

  const { t } = useTranslation();

  return <Stack spacing={2}>
    <TextField
      error={error}
      label={label}
      helperText={error && t('input.numeric.error.range')}
      disabled={disabled}
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
      disabled={disabled}
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
