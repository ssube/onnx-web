import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { PhotoCamera } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import * as React from 'react';

export interface ImageInputProps {
  filter: string;
  image?: Maybe<Blob>;
  label: string;

  hideSelection?: boolean;

  onChange: (file: File) => void;
}

export function ImageInput(props: ImageInputProps) {
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
      return <div>Please select an image.</div>;
    }
  }

  return <Stack direction='row' spacing={2}>
    <Stack>
      <Button component='label' startIcon={<PhotoCamera />}>
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
