import { Card, CardContent, CardMedia, Paper, Stack } from '@mui/material';
import * as React from 'react';

import { ApiResponse } from '../api/client.js';

export interface ImageCardProps {
  value: ApiResponse;
}

export function ImageCard(props: ImageCardProps) {
  const { value } = props;
  const { params, output } = value;

  return <Card sx={{ maxWidth: params.width }}>
    <CardMedia sx={{ height: params.height }}
      component='img'
      image={output}
      title={params.prompt}
    />
    <CardContent>
      <Stack spacing={2}>
        <Paper>CFG: {params.cfg}</Paper>
        <Paper>Steps: {params.steps}</Paper>
        <Paper>Seed: {params.seed}</Paper>
      </Stack>
    </CardContent>
  </Card>;
}
