import { Box, CircularProgress, Stack, Typography } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { STANDARD_SPACING } from '../constants';

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
      spacing={STANDARD_SPACING}
      sx={{ alignItems: 'center' }}
    >
      <CircularProgress />
      <Typography>
        {t('loading.server')}
      </Typography>
    </Stack>
  </Box>;
}
