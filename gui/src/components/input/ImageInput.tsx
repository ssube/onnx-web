import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { PhotoCamera } from '@mui/icons-material';
import { Button, Stack, Typography } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';

import { STANDARD_SPACING } from '../../constants';

export interface ImageInputProps {
  filter: string;
  image?: Maybe<Blob>;
  label: string;

  hideSelection?: boolean;

  onChange: (file: File) => void;
}

export function ImageInput(props: ImageInputProps) {
  const { t } = useTranslation();

  function renderImage() {
    if (doesExist(props.image)) {
      if (mustDefault(props.hideSelection, false)) {
        return undefined;
      }

      return <img
        src={URL.createObjectURL(props.image)}
        style={{
          maxWidth: 512,
          maxHeight: 512,
        }}
      />;
    } else {
      return <Typography>{t('input.image.empty')}</Typography>;
    }
  }

  return <Stack direction='row' spacing={STANDARD_SPACING}>
    <Stack>
      <Button component='label' startIcon={<PhotoCamera />} variant='outlined'>
        {props.label}
        <input
          hidden
          accept={props.filter}
          type='file'
          onChange={(event) => {
            const { files } = event.target;
            if (doesExist(files) && files.length > 0) {
              const file = mustExist(files[0]);
              props.onChange(file);
            }
          }}
        />
      </Button>
    </Stack>
    {renderImage()}
  </Stack>;
}
