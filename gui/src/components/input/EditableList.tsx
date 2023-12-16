import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { OnnxState, StateContext } from '../../state/full.js';

const { useContext, useState, memo, useMemo } = React;

export interface EditableListProps<T> {
  selector: (s: OnnxState) => Array<T>;

  newItem: (l: string, s: string) => T;
  removeItem: (t: T) => void;
  renderItem: (props: {
    model: T;
    onChange: (t: T) => void;
    onRemove: (t: T) => void;
  }) => React.ReactElement;
  setItem: (t: T) => void;
}

export function EditableList<T>(props: EditableListProps<T>) {
  const { newItem, removeItem, renderItem, setItem, selector } = props;

  const { t } = useTranslation();
  const store = mustExist(useContext(StateContext));
  const items = useStore(store, selector);
  const [nextLabel, setNextLabel] = useState('');
  const [nextSource, setNextSource] = useState('');
  const RenderMemo = useMemo(() => memo(renderItem), [renderItem]);

  return <Stack spacing={2}>
    {items.map((model, idx) =>
      <RenderMemo
        key={idx}
        model={model}
        onChange={setItem}
        onRemove={removeItem}
      />
    )}
    <Stack direction='row' spacing={2}>
      <TextField
        label={t('extras.label')}
        variant='outlined'
        value={nextLabel}
        onChange={(event) => setNextLabel(event.target.value)}
      />
      <TextField
        label={t('extras.source')}
        variant='outlined'
        value={nextSource}
        onChange={(event) => setNextSource(event.target.value)}
      />
      <Button onClick={() => {
        setItem(newItem(nextLabel, nextSource));
        setNextLabel('');
        setNextSource('');
      }}>{t('extras.add')}</Button>
    </Stack>
  </Stack>;
}
