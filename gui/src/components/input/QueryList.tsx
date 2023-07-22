import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Alert, FormControl, FormLabel, InputLabel, LinearProgress, MenuItem, Select } from '@mui/material';
import { UseQueryResult } from '@tanstack/react-query';
import * as React from 'react';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export interface QueryListComplete {
  result: UseQueryResult<Array<string>>;
}

export interface QueryListFilter<T> {
  result: UseQueryResult<T>;
  selector: (result: T) => Array<string>;
}

export interface QueryListProps<T> {
  id: string;
  labelKey: string;
  name: string;
  value: string;

  query: QueryListComplete | QueryListFilter<T>;
  showNone?: boolean;

  onChange?: (value: string) => void;
}

export function hasFilter<T>(query: QueryListComplete | QueryListFilter<T>): query is QueryListFilter<T> {
  return Reflect.has(query, 'selector');
}

export function filterQuery<T>(query: QueryListComplete | QueryListFilter<T>, showEmpty: Maybe<string>): Array<string> {
  if (hasFilter(query)) {
    const data = mustExist(query.result.data);
    const selected = (query as QueryListFilter<unknown>).selector(data);
    if (doesExist(showEmpty)) {
      return [showEmpty, ...selected];
    }
    return selected;
  } else {
    const data = Array.from(mustExist(query.result.data));
    if (doesExist(showEmpty)) {
      return [showEmpty, ...data];
    }
    return data;
  }
}

export function QueryList<T>(props: QueryListProps<T>) {
  const { labelKey, query, showNone = false, value } = props;
  const { result } = query;
  const labelID = `query-list-${props.id}-labels`;

  const { t } = useTranslation();

  function firstValidValue(): string {
    if (doesExist(value) && data.includes(value)) {
      return value;
    } else {
      return data[0];
    }
  }

  function noneLabel(): Maybe<string> {
    if (showNone) {
      return 'none';
    }

    return undefined;
  }

  function getLabel(name: string) {
    return mustDefault(t(`${labelKey}.${name}`), name);
  }

  // update state when previous selection was invalid: https://github.com/ssube/onnx-web/issues/120
  useEffect(() => {
    if (result.status === 'success' && doesExist(result.data) && doesExist(props.onChange)) {
      const data = filterQuery(query, noneLabel());
      if (data.includes(value) === false) {
        props.onChange(data[0]);
      }
    }
  }, [result.status]);

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

  // else: success
  const data = filterQuery(query, noneLabel());

  return <FormControl>
    <InputLabel id={labelID}>{props.name}</InputLabel>
    <Select
      labelId={labelID}
      label={props.name}
      value={firstValidValue()}
      onChange={(e) => {
        if (doesExist(props.onChange)) {
          props.onChange(e.target.value);
        }
      }}
    >
      {data.map((name) => <MenuItem key={name} value={name}>{getLabel(name)}</MenuItem>)}
    </Select>
  </FormControl>;
}
