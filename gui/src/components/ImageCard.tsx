import { doesExist } from '@apextoaster/js-utils';
import { Delete, Download } from '@mui/icons-material';
import { Box, Button, Card, CardContent, CardMedia, Grid, Paper } from '@mui/material';
import * as React from 'react';

import { ApiResponse } from '../api/client.js';

export interface ImageCardProps {
  value: ApiResponse;

  onDelete?: (key: ApiResponse) => void;
}

export function GridItem(props: { xs: number; children: React.ReactNode }) {
  return <Grid item xs={props.xs}>
    <Paper elevation={0} sx={{ padding: 1 }}>{props.children}</Paper>
  </Grid>;
}

export function ImageCard(props: ImageCardProps) {
  const { value } = props;
  const { params, output } = value;

  function deleteImage() {
    if (doesExist(props.onDelete)) {
      props.onDelete(value);
    }
  }

  function downloadImage() {
    window.open(output, '_blank');
  }

  return <Card sx={{ maxWidth: params.width }} elevation={2}>
    <CardMedia sx={{ height: params.height }}
      component='img'
      image={output}
      title={params.prompt}
    />
    <CardContent>
      <Box>
        <Grid container spacing={2}>
          <GridItem xs={4}>CFG: {params.cfg}</GridItem>
          <GridItem xs={4}>Steps: {params.steps}</GridItem>
          <GridItem xs={4}>Size: {params.width}x{params.height}</GridItem>
          <GridItem xs={4}>Seed: {params.seed}</GridItem>
          <GridItem xs={8}>Scheduler: {params.scheduler}</GridItem>
          <GridItem xs={12}>{params.prompt}</GridItem>
          <GridItem xs={2}>
            <Button onClick={downloadImage}>
              <Download />
            </Button>
          </GridItem>
          <GridItem xs={2}>
            <Button onClick={deleteImage}>
              <Delete />
            </Button>
          </GridItem>
        </Grid>
      </Box>
    </CardContent>
  </Card>;
}
