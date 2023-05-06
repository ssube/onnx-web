import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';

const { useState } = React;

export interface EditableListProps<T> {
  items: Array<T>;

  newItem: (l: string, s: string) => T;
  renderItem: (t: T) => React.ReactElement;
  setItems: (ts: Array<T>) => void;
}

export function EditableList<T>(props: EditableListProps<T>) {
  const { items, newItem, renderItem, setItems } = props;
  const [nextLabel, setNextLabel] = useState('');
  const [nextSource, setNextSource] = useState('');

  return <Stack spacing={2}>
    {items.map((it, idx) => <Stack direction='row' key={idx} spacing={2}>
      {renderItem(it)}
      <Button onClick={() => setItems([
        ...items.slice(0, idx),
        ...items.slice(idx + 1, items.length),
      ])}>Remove</Button>
    </Stack>)}
    <Stack direction='row' spacing={2}>
      <TextField
        label='Label'
        variant='outlined'
        value={nextLabel}
        onChange={(event) => setNextLabel(event.target.value)}
      />
      <TextField
        label='Source'
        variant='outlined'
        value={nextSource}
        onChange={(event) => setNextSource(event.target.value)}
      />
      <Button onClick={() => {
        setItems([...items, newItem(nextLabel, nextSource)]);
        setNextLabel('');
      }}>New</Button>
    </Stack>
  </Stack>;
}
