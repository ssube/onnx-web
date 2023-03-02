import { Link, Typography } from '@mui/material';
import * as React from 'react';

export function Logo() {
  return <Typography variant='h3' gutterBottom>
    <Link href='https://github.com/ssube/onnx-web' target='_blank' underline='hover'>ONNX Web</Link>
  </Typography>;
}
