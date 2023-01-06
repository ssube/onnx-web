import { Card, CardContent, CardMedia, Paper, Stack } from '@mui/material';
import * as React from 'react';
import { UseMutationResult } from 'react-query';

import { ApiResponse } from '../api/client.js';

export interface ImageCardProps {
  result: UseMutationResult<ApiResponse, unknown, void>;
}

export function ImageCard(props: ImageCardProps) {
  const { result } = props;

  if (result.status === 'error') {
    if (result.error instanceof Error) {
      return <div>{result.error.message}</div>;
    } else {
      return <div>Unknown error generating image.</div>;
    }
  }

  if (result.status === 'loading') {
    return <div>Generating...</div>;
  }

  if (result.status === 'success') {
    const { data } = result;
    const { params, output } = data;

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

  return <div>No result. Press Generate.</div>;
}
