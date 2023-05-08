import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useStore } from 'zustand';

import { OnnxState, StateContext } from '../../state';

const { useContext, useState, memo, useMemo } = React;

export interface EditableListProps<T> {
  // items: Array<T>;
  selector: (s: OnnxState) => Array<T>;

  newItem: (l: string, s: string) => T;
  // removeItem: (t: T) => void;
  renderItem: (props: {
    model: T;
    onChange: (t: T) => void;
  }) => React.ReactElement;
  setItem: (t: T) => void;
}

export function EditableList<T>(props: EditableListProps<T>) {
  const { newItem, renderItem, setItem, selector } = props;

  const state = mustExist(useContext(StateContext));
  const items = useStore(state, selector);
  const [nextLabel, setNextLabel] = useState('');
  const [nextSource, setNextSource] = useState('');
  const RenderMemo = useMemo(() => memo(renderItem), [renderItem]);

  return <Stack spacing={2}>
    {items.map((model, idx) => <Stack direction='row' key={idx} spacing={2}>
      <RenderMemo
        model={model}
        onChange={setItem}
      />
      <Button onClick={() => {
        // removeItem(model);
      }}>Remove</Button>
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
        setItem(newItem(nextLabel, nextSource));
        setNextLabel('');
        setNextSource('');
      }}>New</Button>
    </Stack>
  </Stack>;
}
