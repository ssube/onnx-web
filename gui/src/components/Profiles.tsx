import * as React from 'react';
import { useContext } from 'react';
import { doesExist, mustExist } from '@apextoaster/js-utils';
import { useStore } from 'zustand';
import { useTranslation } from 'react-i18next';
import {
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  ListItem,
  ListItemText,
  Autocomplete,
  Stack,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

import { StateContext } from '../state.js';
import { BaseImgParams } from '../client/types.js';

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
