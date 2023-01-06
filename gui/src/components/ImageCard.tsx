import { Box, Card, CardContent, CardMedia, Grid, Paper } from '@mui/material';
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
      <Box>
        <Grid container spacing={2}>
          <Grid item xs={4}>
            <Paper>CFG: {params.cfg}</Paper>
          </Grid>
          <Grid item xs={4}>
            <Paper>Steps: {params.steps}</Paper>
          </Grid>
          <Grid item xs={4}>
            <Paper>Size: {params.width}x{params.height}</Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper>Seed: {params.seed}</Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper>Scheduler: {params.scheduler}</Paper>
          </Grid>
          <Grid item xs={12}>
            <Paper>{params.prompt}</Paper>
          </Grid>
        </Grid>
      </Box>
    </CardContent>
  </Card>;
}
