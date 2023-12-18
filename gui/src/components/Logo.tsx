import { Help } from '@mui/icons-material';
import { IconButton, Link, Typography } from '@mui/material';
import * as React from 'react';

export const URL_DOCS = 'https://www.onnx-web.ai/docs';
export const URL_REPO = 'https://github.com/ssube/onnx-web';
export const NEW_WINDOW = '_blank';

export function openDocSite() {
  window.open(URL_DOCS, NEW_WINDOW);
}

export function Logo() {
  return <Typography variant='h3' gutterBottom>
    <Link href={URL_REPO} target={NEW_WINDOW} underline='hover'>ONNX Web</Link>
    <IconButton onClick={openDocSite}>
      <Help />
    </IconButton>
  </Typography>;
}
