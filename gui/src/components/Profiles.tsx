import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Delete as DeleteIcon, ImageSearch, Save as SaveIcon } from '@mui/icons-material';
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
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { BaseImgParams, Txt2ImgParams } from '../client/types.js';
import { StateContext } from '../state.js';

export interface ProfilesProps {
  params: BaseImgParams;
  setParams: ((params: BaseImgParams) => void) | undefined;
}

export function Profiles(props: ProfilesProps) {
  const state = mustExist(useContext(StateContext));

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const saveProfile = useStore(state, (s) => s.saveProfile);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const removeProfile = useStore(state, (s) => s.removeProfile);
  const profiles = useStore(state, (s) => s.profiles);
  const highres = useStore(state, (s) => s.highres);
  const upscale = useStore(state, (s) => s.upscale);
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const [profileName, setProfileName] = React.useState('');
  const { t } = useTranslation();

  return <>
    <Autocomplete
      id="profile-select"
      options={profiles}
      sx={{ width: 200 }}
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
          <Button component='label' variant="contained">
            <ImageSearch />
            <input
              hidden
              accept={'*.json,*.jpg,*.jpeg,*.png,*.txt'}
              type='file'
              onChange={(event) => {
                const { files } = event.target;
                if (doesExist(files) && files.length > 0) {
                  const file = mustExist(files[0]);
                  // eslint-disable-next-line @typescript-eslint/no-floating-promises
                  loadParamsFromFile(file).then((newParams) => {
                    if (doesExist(props.setParams) && doesExist(newParams)) {
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
        </Stack>
      )}
      onChange={(event, value) => {
        if (doesExist(value) && doesExist(props.setParams)) {
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
              highResParams: highres,
              upscaleParams: upscale,
            });
            setDialogOpen(false);
            setProfileName('');
          }}
        >{t('profile.save')}</Button>
      </DialogActions>
    </Dialog>
  </>;
}

export async function loadParamsFromFile(file: File): Promise<Partial<Txt2ImgParams>> {
  const parts = file.name.toLocaleLowerCase().split('.');
  const ext = parts[parts.length - 1];

  switch (ext) {
    case 'jpg':
    case 'jpeg':
    case 'png':
      return parseImageParams(file);
    case 'json':
    {
      const params = JSON.parse(await file.text());
      return params.params as Txt2ImgParams;
    }
    case 'txt':
    default:
      return parseAutoComment(await file.text());
  }
}

export async function parseImageParams(file: File): Promise<Partial<Txt2ImgParams>> {
  const tags = await ExifReader.load(file);

  // handle lowercase variation from my earlier mistakes
  const makerNote = tags.MakerNote || tags['maker note'];
  // eslint-disable-next-line dot-notation, @typescript-eslint/strict-boolean-expressions
  const userComment = tags.UserComment || tags['Parameters'] || tags['parameters'];

  if (doesExist(makerNote) && isProbablyJSON(makerNote.value)) {
    const params = JSON.parse(makerNote.value);
    return params.params as Txt2ImgParams; // TODO: enforce schema and do some error handling
  }

  if (doesExist(userComment) && typeof userComment.value === 'string') {
    return parseAutoComment(userComment.value);
  }

  return {};
}

export function isProbablyJSON(maybeJSON: unknown): maybeJSON is string {
  return typeof maybeJSON === 'string' && maybeJSON[0] === '{' && maybeJSON[maybeJSON.length - 1] === '}';
}

export const NEGATIVE_PROMPT_TAG = 'Negative prompt:';

export function parseAutoComment(comment: string): Partial<Txt2ImgParams> {
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
