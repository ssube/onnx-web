import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import * as React from 'react';
import { UseQueryResult } from 'react-query';

export interface QueryListComplete {
  result: UseQueryResult<Array<string>>;
}

export interface QueryListFilter<T> {
  result: UseQueryResult<T>;
  selector: (result: T) => Array<string>;
}

export interface QueryListProps<T> {
  id: string;
  labels: Record<string, string>;
  name: string;
  value: string;

  query: QueryListComplete | QueryListFilter<T>;

  onChange?: (value: string) => void;
}

export function hasFilter<T>(query: QueryListComplete | QueryListFilter<T>): query is QueryListFilter<T> {
  return Reflect.has(query, 'selector');
}

export function filterQuery<T>(query: QueryListComplete | QueryListFilter<T>): Array<string> {
  if (hasFilter(query)) {
    const data = mustExist(query.result.data);
    return (query as QueryListFilter<unknown>).selector(data);
  } else {
    return mustExist(query.result.data);
  }
}

export function QueryList<T>(props: QueryListProps<T>) {
  const { labels, query, value } = props;
  const { result } = query;

  if (result.status === 'error') {
    if (result.error instanceof Error) {
      return <div>Error: {result.error.message}</div>;
    } else {
      return <div>Unknown Error</div>;
    }
  }

  if (result.status === 'loading') {
    return <div>Loading...</div>;
  }

  if (result.status === 'idle') {
    return <div>Idle?</div>;
  }

  // else: success
  const labelID = `query-list-${props.id}-labels`;
  const data = filterQuery(query);

  return <FormControl>
    <InputLabel id={labelID}>{props.name}</InputLabel>
    <Select
      labelId={labelID}
      label={props.name}
      value={value}
      onChange={(e) => {
        if (doesExist(props.onChange)) {
          props.onChange(e.target.value);
        }
      }}
    >
      {data.map((name) => <MenuItem key={name} value={name}>{mustDefault(labels[name], name)}</MenuItem>)}
    </Select>
  </FormControl>;
}
