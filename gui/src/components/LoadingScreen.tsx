import { Box, CircularProgress, Stack } from '@mui/material';
import * as React from 'react';

export function LoadingScreen() {
  return <Box sx={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 512,
  }}>
    <Stack
      direction='column'
      spacing={2}
      sx={{ alignItems: 'center' }}
    >
      <CircularProgress />
    </Stack>
  </Box>;
}
