import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { PhotoCamera } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import * as React from 'react';

export interface ImageInputProps {
  filter: string;
  hidden?: boolean;
  image?: Maybe<Blob>;
  label: string;

  onChange: (file: File) => void;
  renderImage?: (image: Maybe<Blob>) => React.ReactNode;
}

export function ImageInput(props: ImageInputProps) {
  function renderImage() {
    if (mustDefault(props.hidden, false)) {
      return undefined;
    }

    if (doesExist(props.renderImage)) {
      return props.renderImage(props.image);
    }

    if (doesExist(props.image)) {
      return <img src={URL.createObjectURL(props.image)} />;
    } else {
      return <div>Please select an image.</div>;
    }
  }

  return <Stack direction='row' spacing={2}>
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
    {renderImage()}
  </Stack>;
}
