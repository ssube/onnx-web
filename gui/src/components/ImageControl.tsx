import { doesExist } from '@apextoaster/js-utils';
import { Container, Stack, TextField } from '@mui/material';
import * as React from 'react';

export interface ImageParams {
  cfg: number;
  steps: number;
  width: number;
  height: number;
}

export interface ImageControlProps {
  params: ImageParams;
  onChange?: (params: ImageParams) => void;
}

export function ImageControl(props: ImageControlProps) {
  const { params } = props;

  return <Stack spacing={2}>
    <Stack direction="row" spacing={4}>
      <TextField
        label="CFG"
        variant="outlined"
        type="number"
        inputProps={{ min: 0, max: 30, step: 1 }}
        value={params.cfg}
        onChange={(event) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              cfg: parseInt(event.target.value, 10),
            });
          }
        }}
      />
      <TextField
        label="Steps"
        variant="outlined"
        type="number"
        inputProps={{ min: 1, max: 150, step: 1 }}
        value={params.steps}
        onChange={(event) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              steps: parseInt(event.target.value, 10),
            });
          }
        }}
      />
    </Stack>
    <Stack direction="row" spacing={4}>
      <TextField
        label="Width"
        variant="outlined"
        type="number"
        inputProps={{ min: 1, max: 512, step: 16 }}
        value={params.width}
        onChange={(event) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              width: parseInt(event.target.value, 10),
            });
          }
        }}
      />
      <TextField
        label="Height"
        variant="outlined"
        type="number"
        inputProps={{ min: 1, max: 512, step: 16 }}
        value={params.height}
        onChange={(event) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              height: parseInt(event.target.value, 10),
            });
          }
        }}
      />
    </Stack>
  </Stack>;
}