import { Grid } from '@mui/material';
import { useState } from 'react';
import * as React from 'react';
import { UseMutationResult } from 'react-query';

export interface MutationHistoryProps<T> {
  element: React.ComponentType<{value: T}>;
  limit: number;
  result: UseMutationResult<T, unknown, void>;

  isPresent: (list: Array<T>, item: T) => boolean;
}

export function MutationHistory<T>(props: MutationHistoryProps<T>) {
  const { limit, result } = props;
  const { status } = result;

  const [history, setHistory] = useState<Array<T>>([]);
  const children = [];

  if (status === 'loading') {
    children.push(<div>Generating...</div>);
  }

  if (status === 'success') {
    const { data } = result;
    if (props.isPresent(history, data)) {
      // item already exists, skip it
    } else {
      setHistory([
        data,
        ...history,
      ].slice(0, limit));
    }
  }

  if (history.length > 0) {
    children.push(...history.map((item) => <props.element value={item} />));
  } else {
    // only show the prompt when the button has not been pushed
    if (status !== 'loading') {
      children.push(<div>No results. Press Generate.</div>);
    }
  }

  return <Grid container spacing={2}>{children.slice(0, limit).map((child) => <Grid item xs={6}>{child}</Grid>)}</Grid>;
}
