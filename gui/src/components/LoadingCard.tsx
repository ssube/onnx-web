import { Card, CardContent, CircularProgress } from '@mui/material';
import * as React from 'react';

export interface LoadingCardProps {
  height: number;
  width: number;
}

export function LoadingCard(props: LoadingCardProps) {
  return <Card sx={{ maxWidth: props.width }}>
    <CardContent sx={{ height: props.height }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: props.height }}>
        <CircularProgress />
      </div>
    </CardContent>
  </Card>;
}
