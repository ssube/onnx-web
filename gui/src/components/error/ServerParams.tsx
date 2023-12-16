import { Alert, AlertTitle, Typography } from '@mui/material';
import * as React from 'react';

export interface ServerParamsErrorProps {
  error: unknown;
  root: string;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'unknown error';
}

export function ServerParamsError(props: ServerParamsErrorProps) {
  return <Alert severity='error'>
    <AlertTitle>
      Error fetching server parameters
    </AlertTitle>
    <Typography>
      Could not fetch parameters from the onnx-web API server at <code>{props.root}</code>.
    </Typography>
    <Typography>
      {getErrorMessage(props.error)}
    </Typography>
  </Alert>;
}
