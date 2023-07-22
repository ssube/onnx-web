import { InvalidArgumentError, Maybe, doesExist, mustExist } from '@apextoaster/js-utils';
import { Delete as DeleteIcon, Download, ImageSearch, Save as SaveIcon } from '@mui/icons-material';
import {
  Autocomplete,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  ListItem,
  ListItemText,
  Stack,
  TextField,
} from '@mui/material';
import * as ExifReader from 'exifreader';
import { defaultTo, isString } from 'lodash';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { BaseImgParams, HighresParams, Txt2ImgParams, UpscaleParams } from '../client/types.js';
import { StateContext } from '../state.js';

const { useState, Fragment } = React;

export interface ProfilesProps {
  highres: HighresParams;
  params: BaseImgParams;
  upscale: UpscaleParams;

  setHighres(params: HighresParams): void;
  setParams(params: BaseImgParams): void;
  setUpscale(params: UpscaleParams): void;
}

export function Profiles(props: ProfilesProps) {
  const state = mustExist(useContext(StateContext));
  const profiles = useStore(state, (s) => s.profiles);

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const saveProfile = useStore(state, (s) => s.saveProfile);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeProfile = useStore(state, (s) => s.removeProfile);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [profileName, setProfileName] = useState('');
  const { t } = useTranslation();

  return <Stack direction='row' spacing={2}>
    <Autocomplete
      id="profile-select"
      options={profiles}
      sx={{ width: '25em' }}
      getOptionLabel={(option) => option.name}
      clearOnBlur
      renderOption={(optionProps, option) => (
        <ListItem
          {...optionProps}
          secondaryAction={
            <IconButton edge="end" onClick={(event) => {
              event.preventDefault();
              removeProfile(option.name);
            }}>
              <DeleteIcon />
            </IconButton>
          }
        >
          <ListItemText primary={option.name} />
        </ListItem>
      )}
      renderInput={(params) => (
        <Stack direction="row">
          <TextField
            {...params}
            label={t('profile.load')}
            inputProps={{
              ...params.inputProps,
              autoComplete: 'new-password', // disable autocomplete and autofill
            }}
          />
          <Button type="button" variant="contained" onClick={() => setDialogOpen(true)}>
            <SaveIcon />
          </Button>
        </Stack>
      )}
      onChange={(event, value) => {
        if (doesExist(value)) {
          props.setParams({
            ...value.params
          });
        }
      }}
    />
    <Dialog
      open={dialogOpen}
      onClose={() => setDialogOpen(false)}
    >
      <DialogTitle>{t('profile.saveProfile')}</DialogTitle>
      <DialogContent>
        <TextField
          variant="standard"
          label={t('profile.name')}
          value={profileName}
          onChange={(event) => setProfileName(event.target.value)}
          fullWidth
        />
      </DialogContent>
      <DialogActions>
        <Button
          variant='contained'
          onClick={() => setDialogOpen(false)}
        >{t('profile.cancel')}</Button>
        <Button
          variant='contained'
          onClick={() => {
            saveProfile({
              params: props.params,
              name: profileName,
              highResParams: props.highres,
              upscaleParams: props.upscale,
            });
            setDialogOpen(false);
            setProfileName('');
          }}
        >{t('profile.save')}</Button>
      </DialogActions>
    </Dialog>
    <Button component='label' variant="contained">
      <ImageSearch />
      <input
        hidden
        accept={'.json,.jpg,.jpeg,.png,.txt,.webp'}
        type='file'
        onChange={(event) => {
          const { files } = event.target;
          if (doesExist(files) && files.length > 0) {
            const file = mustExist(files[0]);
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            loadParamsFromFile(file).then((newParams) => {
              if (doesExist(newParams)) {
                props.setParams({
                  ...props.params,
                  ...newParams,
                });
              }
            });
          }
        }}
        onClick={(event) => {
          event.currentTarget.value = '';
        }}
      />
    </Button>
    <Button component='label' variant='contained' onClick={() => {
      downloadParamsAsFile(props.params);
    }}>
      <Download />
    </Button>
  </Stack>;
}

export async function loadParamsFromFile(file: File): Promise<Partial<Txt2ImgParams>> {
  const parts = file.name.toLocaleLowerCase().split('.');
  const ext = parts[parts.length - 1];

  switch (ext) {
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'webp':
      return parseImageParams(file);
    case 'json':
      return parseJSONParams(await file.text());
    case 'txt':
    default:
      return parseAutoComment(await file.text());
  }
}

/**
 * from https://stackoverflow.com/a/30800715
 */
export function downloadParamsAsFile(params: Txt2ImgParams): void {
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify({
    params,
  }));
  const elem = document.createElement('a');
  elem.setAttribute('href', dataStr);
  elem.setAttribute('download', 'parameters.json');
  document.body.appendChild(elem); // required for firefox
  elem.click();
  elem.remove();
}

export async function parseImageParams(file: File): Promise<Partial<Txt2ImgParams>> {
  const tags = await ExifReader.load(file);

  // handle lowercase variation from my earlier mistakes
  const makerNote = decodeTag(defaultTo(tags.MakerNote, tags['maker note']));
  // eslint-disable-next-line dot-notation, @typescript-eslint/strict-boolean-expressions
  const userComment = decodeTag(defaultTo(defaultTo(tags.UserComment, tags['Parameters']), tags['parameters']));

  if (doesExist(makerNote) && isProbablyJSON(makerNote)) {
    return parseJSONParams(makerNote);
  }

  if (doesExist(userComment)) {
    return parseAutoComment(userComment);
  }

  return {};
}

export function isNumberArray(it: unknown): it is Array<number> {
  return Array.isArray(it) && typeof it[0] === 'number';
}

export function decodeTag(tag: Maybe<ExifReader.XmpTag | (ExifReader.NumberTag & ExifReader.NumberArrayTag)>): Maybe<string> {
  // eslint-disable-next-line no-restricted-syntax
  if (!doesExist(tag)) {
    return undefined;
  }

  if (isString(tag.value)) {
    return tag.value;
  }

  if (tag.description === '[Unicode encoded text]' && isNumberArray(tag.value)) {
    return Buffer.from(tag.value).toString('utf-8');
  }

  throw new InvalidArgumentError('tag value cannot be decoded');
}

export async function parseJSONParams(json: string): Promise<Partial<Txt2ImgParams>> {
  const data = JSON.parse(json);
  const params: Partial<Txt2ImgParams> = {
    ...data.params,
  };

  const size = defaultTo(data.input_size, data.size);
  if (doesExist(size)) {
    params.height = size.height;
    params.width = size.width;
  }

  return params;
}

export function isProbablyJSON(maybeJSON: unknown): boolean {
  return typeof maybeJSON === 'string' && maybeJSON[0] === '{' && maybeJSON[maybeJSON.length - 1] === '}';
}

export const NEGATIVE_PROMPT_TAG = 'Negative prompt:';

export async function parseAutoComment(comment: string): Promise<Partial<Txt2ImgParams>> {
  if (isProbablyJSON(comment)) {
    return parseJSONParams(comment);
  }

  const lines = comment.split('\n');
  const [prompt, maybeNegative, ...otherLines] = lines;

  const params: Partial<Txt2ImgParams> = {
    prompt,
  };

  // check if maybeNegative is the negative prompt
  if (maybeNegative.startsWith(NEGATIVE_PROMPT_TAG)) {
    params.negativePrompt = maybeNegative.substring(NEGATIVE_PROMPT_TAG.length).trim();
  } else {
    otherLines.unshift(maybeNegative);
  }

  // join rest and split on commas
  const other = otherLines.join(' ');
  const otherParams = other.split(',');

  for (const param of otherParams) {
    const [key, value] = param.split(':');

    switch (key.toLocaleLowerCase().trim()) {
      case 'steps':
        params.steps = parseInt(value, 10);
        break;
      case 'sampler':
        params.scheduler = value;
        break;
      case 'cfg scale':
        params.cfg = parseInt(value, 10);
        break;
      case 'seed':
        params.seed = parseInt(value, 10);
        break;
      case 'size':
        {
          const [width, height] = value.split('x');
          params.height = parseInt(height, 10);
          params.width = parseInt(width, 10);
        }
        break;
      default:
      // unknown param
    }
  }

  return params;
}
