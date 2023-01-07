import { Grid } from '@mui/material';
import { useState } from 'react';
import * as React from 'react';
import { UseMutationResult } from 'react-query';
import { LoadingCard } from './LoadingCard';

export interface MutationHistoryChildProps<T> {
  value: T;

  onDelete: (key: T) => void;
}

export interface MutationHistoryProps<T> {
  element: React.ComponentType<MutationHistoryChildProps<T>>;
  limit: number;
  result: UseMutationResult<T, unknown, void>;

  isEqual: (a: T, b: T) => boolean;
}

export function MutationHistory<T>(props: MutationHistoryProps<T>) {
  const { limit, result } = props;
  const { status } = result;

  const [history, setHistory] = useState<Array<T>>([]);
  const children = [];

  if (status === 'loading') {
    children.push(<LoadingCard height={512} width={512} />); // TODO: get dimensions from parent
  }

  if (status === 'success') {
    const { data } = result;
    if (history.some((other) => props.isEqual(data, other))) {
      // item already exists, skip it
    } else {
      setHistory([
        data,
        ...history,
      ].slice(0, limit));
    }
  }

  function removeHistory(data: T) {
    setHistory(history.filter((item) => props.isEqual(item, data) === false));
  }

  if (history.length > 0) {
    children.push(...history.map((item) => <props.element value={item} onDelete={removeHistory} />));
  } else {
    // only show the prompt when the button has not been pushed
    if (status !== 'loading') {
      children.push(<div>No results. Press Generate.</div>);
    }
  }

  return <Grid container spacing={2}>{children.slice(0, limit).map((child, idx) => <Grid item key={idx} xs={6}>{child}</Grid>)}</Grid>;
}
