import { Alert, AlertTitle, Typography } from '@mui/material';
import * as React from 'react';

import { PARAM_VERSION } from '../../config.js';

export interface ParamsVersionErrorProps {
  root: string;
  version: string;
}

export function ParamsVersionError(props: ParamsVersionErrorProps) {
  return <Alert severity='error'>
    <AlertTitle>
      Parameter version error
    </AlertTitle>
    <Typography>
      The server returned parameters that are too old for the client to load.
    </Typography>
    <Typography>
      The server parameters are version <code>{props.version}</code>, but this client's required version
      is <code>{PARAM_VERSION}</code>. Please update your server or use a matching version of the client.
    </Typography>
  </Alert>;
}

