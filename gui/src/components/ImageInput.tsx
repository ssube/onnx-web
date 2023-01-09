import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { PhotoCamera } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import * as React from 'react';

const { useState } = React;

export interface ImageInputProps {
  filter: string;
  hidden?: boolean;
  label: string;

  onChange: (file: File) => void;
  renderImage?: (image: string | undefined) => React.ReactNode;
}

export function ImageInput(props: ImageInputProps) {
  const [image, setImage] = useState<string>();

  function renderImage() {
    if (mustDefault(props.hidden, false)) {
      return undefined;
    }

    if (doesExist(props.renderImage)) {
      return props.renderImage(image);
    }

    return <img src={image} />;
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

            if (doesExist(image)) {
              URL.revokeObjectURL(image);
            }

            setImage(URL.createObjectURL(file));
            props.onChange(file);
          }
        }}
      />
    </Button>
    {renderImage()}
  </Stack>;
}
