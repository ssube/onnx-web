import React, { useState } from 'react';
import { Alert, Collapse, IconButton } from '@mui/material';
import { Close } from '@mui/icons-material';
import { useTranslation } from 'react-i18next';

export function Motd() {
  const [open, setOpen] = useState(true);
  const { t } = useTranslation();

  return <Collapse in={open}>
    <Alert
      action={
        <IconButton
          aria-label="close"
          color="inherit"
          size="small"
          onClick={() => {
            setOpen(false);
          }}
        >
          <Close fontSize="inherit" />
        </IconButton>
      }
      severity='info'
      sx={{ mb: 2 }}
    >
      {t('motd')}
    </Alert>
  </Collapse>;
}
