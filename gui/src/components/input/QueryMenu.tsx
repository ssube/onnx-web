import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { KeyboardArrowDown } from '@mui/icons-material';
import { Alert, Box, Button, FormControl, FormLabel, LinearProgress, Menu, MenuItem, Typography } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { UseQueryResult } from 'react-query';

export interface QueryMenuComplete {
  result: UseQueryResult<Array<string>>;
}

export interface QueryMenuFilter<T> {
  result: UseQueryResult<T>;
  selector: (result: T) => Array<string>;
}

export interface QueryMenuProps<T> {
  id: string;
  labelKey: string;
  name: string;

  query: QueryMenuComplete | QueryMenuFilter<T>;
  showEmpty?: boolean;

  onSelect?: (value: string) => void;
}

export function hasFilter<T>(query: QueryMenuComplete | QueryMenuFilter<T>): query is QueryMenuFilter<T> {
  return Reflect.has(query, 'selector');
}

export function filterQuery<T>(query: QueryMenuComplete | QueryMenuFilter<T>, showEmpty: boolean): Array<string> {
  if (hasFilter(query)) {
    const data = mustExist(query.result.data);
    const selected = (query as QueryMenuFilter<unknown>).selector(data);
    if (showEmpty) {
      return ['', ...selected];
    }
    return selected;
  } else {
    const data = Array.from(mustExist(query.result.data));
    if (showEmpty) {
      return ['', ...data];
    }
    return data;
  }
}

export function QueryMenu<T>(props: QueryMenuProps<T>) {
  const { id, labelKey, name, query, showEmpty = false } = props;
  const { result } = query;
  const labelID = `query-menu-${props.id}-labels`;

  const { t } = useTranslation();

  const [anchor, setAnchor] = React.useState<Maybe<HTMLElement>>(undefined);

  function closeMenu() {
    setAnchor(undefined);
  }

  function openMenu(event: React.MouseEvent<HTMLButtonElement>) {
    setAnchor(event.currentTarget);
  }

  function selectItem(value: string) {
    closeMenu();
    if (doesExist(props.onSelect)) {
      props.onSelect(value);
    }
  }

  function getLabel(key: string) {
    return mustDefault(t(`${labelKey}.${key}`), key);
  }

  if (result.status === 'error') {
    if (result.error instanceof Error) {
      return <Alert severity='error'>{t('input.list.error.specific', {
        message: result.error.message,
      })}</Alert>;
    } else {
      return <Alert severity='error'>{t('input.list.error.unknown')}</Alert>;
    }
  }

  if (result.status === 'loading') {
    return <FormControl>
      <FormLabel id={labelID}>{props.name}</FormLabel>
      <LinearProgress />
    </FormControl>;
  }

  if (result.status === 'idle') {
    return <Typography>{t('input.list.idle')}</Typography>;
  }

  // else: success
  const data = filterQuery(query, showEmpty);

  return <Box>
    <Button
      id={`${id}-button`}
      onClick={openMenu}
      endIcon={<KeyboardArrowDown />}
      variant='outlined'
    >
      {name}
    </Button>
    <Menu
      id={`${id}-menu`}
      anchorEl={anchor}
      open={doesExist(anchor)}
      onClose={closeMenu}
      MenuListProps={{
        'aria-labelledby': `${id}-button`,
      }}
    >
      {data.map((it, idx) => <MenuItem key={it} onClick={() => selectItem(it)}>{getLabel(it)}</MenuItem>)}
    </Menu>
  </Box>;
}

