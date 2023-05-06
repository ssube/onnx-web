import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';

const { useState } = React;

export interface EditableListProps<T> {
  items: Array<T>;

  newItem: (s: string) => T;
  renderItem: (t: T) => React.ReactElement;
  setItems: (ts: Array<T>) => void;
}

export function EditableList<T>(props: EditableListProps<T>) {
  const { items, newItem, renderItem, setItems } = props;
  const [nextItem, setNextItem] = useState('');

  return <Stack>
    {items.map((it, idx) => <Stack direction='row' key={idx}>
      {renderItem(it)}
      <Button onClick={() => setItems([
        ...items.slice(0, idx),
        ...items.slice(idx + 1, items.length),
      ])}>Remove</Button>
    </Stack>)}
    <Stack direction='row'>
      <TextField
        label='Source'
        variant='outlined'
        value={nextItem}
        onChange={(event) => setNextItem(event.target.value)}
      />
      <Button onClick={() => {
        setItems([...items, newItem(nextItem)]);
        setNextItem('');
      }}>New</Button>
    </Stack>
  </Stack>;
}
