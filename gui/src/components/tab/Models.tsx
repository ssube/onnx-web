import { mustExist } from '@apextoaster/js-utils';
import { Accordion, AccordionDetails, AccordionSummary, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useStore } from 'zustand';

import { StateContext } from '../../state.js';
import { EditableList } from '../input/EditableList';

export function Models() {
  const state = mustExist(React.useContext(StateContext));
  const extras = useStore(state, (s) => s.extras);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setExtras = useStore(state, (s) => s.setExtras);

  return <Stack>
    <Accordion>
      <AccordionSummary>
        Diffusion Models
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          items={extras.diffusion}
          newItem={(s) => s}
          renderItem={(t) => <div key={t}>{t}</div>}
          setItems={(diffusion) => setExtras({
            ...extras,
            diffusion,
          })}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Correction Models
      </AccordionSummary>
      <AccordionDetails>
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Upscaling Models
      </AccordionSummary>
      <AccordionDetails>
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Additional Networks
      </AccordionSummary>
      <AccordionDetails>
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Other Sources
      </AccordionSummary>
      <AccordionDetails>
      </AccordionDetails>
    </Accordion>
    <Button color='warning'>Save & Convert</Button>
  </Stack>;
}
