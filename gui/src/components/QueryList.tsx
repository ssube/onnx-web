import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import * as React from 'react';
import { UseQueryResult } from 'react-query';

export interface QueryListProps {
  labels: Record<string, string>;
  name: string;
  result: UseQueryResult<Array<string>>;
  value: string;

  onChange?: (value: string) => void;
}

export function QueryList(props: QueryListProps) {
  const { labels, result, value } = props;

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
  const data = mustExist(result.data);
  return <FormControl>
    <InputLabel>{props.name}</InputLabel>
    <Select value={value} onChange={(e) => {
      if (doesExist(props.onChange)) {
        props.onChange(e.target.value);
      }
    }}>
      {data.map((name) => <MenuItem key={name} value={name}>{mustDefault(labels[name], name)}</MenuItem>)}
    </Select>
  </FormControl>;
}
