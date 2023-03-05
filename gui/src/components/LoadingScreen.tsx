import { Box, CircularProgress, Stack, Typography } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

export function LoadingScreen() {
  const { t } = useTranslation();

  return <Box sx={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: window.innerHeight,
  }}>
    <Stack
      direction='column'
      spacing={2}
      sx={{ alignItems: 'center' }}
    >
      <CircularProgress />
      <Typography>
        {t('loading.server')}
      </Typography>
    </Stack>
  </Box>;
}
