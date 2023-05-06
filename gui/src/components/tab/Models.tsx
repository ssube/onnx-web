import { mustExist } from '@apextoaster/js-utils';
import { Accordion, AccordionDetails, AccordionSummary, Button, Stack } from '@mui/material';
import * as React from 'react';
import _ from 'lodash';
import { useStore } from 'zustand';

import { StateContext } from '../../state.js';
import { EditableList } from '../input/EditableList';
import { DiffusionModelInput } from '../input/model/DiffusionModel.js';
import { SafetensorFormat } from '../../types.js';
import { CorrectionModelInput } from '../input/model/CorrectionModel.js';
import { UpscalingModelInput } from '../input/model/UpscalingModel.js';
import { ExtraSourceInput } from '../input/model/ExtraSource.js';
import { ExtraNetworkInput } from '../input/model/ExtraNetwork.js';
// eslint-disable-next-line @typescript-eslint/unbound-method
const { kebabCase }  = _;

export function Models() {
  const state = mustExist(React.useContext(StateContext));
  const extras = useStore(state, (s) => s.extras);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setExtras = useStore(state, (s) => s.setExtras);

  return <Stack spacing={2}>
    <Accordion>
      <AccordionSummary>
        Diffusion Models
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          items={extras.diffusion}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: kebabCase(l),
            source: s,
          })}
          renderItem={(t) => <DiffusionModelInput model={t}/>}
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
        <EditableList
          items={extras.correction}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: kebabCase(l),
            source: s,
          })}
          renderItem={(t) => <CorrectionModelInput model={t}/>}
          setItems={(correction) => setExtras({
            ...extras,
            correction,
          })}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Upscaling Models
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          items={extras.upscaling}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: kebabCase(l),
            scale: 4,
            source: s,
          })}
          renderItem={(t) => <UpscalingModelInput model={t}/>}
          setItems={(upscaling) => setExtras({
            ...extras,
            upscaling,
          })}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Extra Networks
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          items={extras.networks}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            model: 'embeddings' as const,
            name: kebabCase(l),
            source: s,
            type: 'inversion' as const,
          })}
          renderItem={(t) => <ExtraNetworkInput model={t}/>}
          setItems={(networks) => setExtras({
            ...extras,
            networks,
          })}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        Other Sources
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          items={extras.sources}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: kebabCase(l),
            source: s,
          })}
          renderItem={(t) => <ExtraSourceInput model={t}/>}
          setItems={(sources) => setExtras({
            ...extras,
            sources,
          })}
        />
      </AccordionDetails>
    </Accordion>
    <Button color='warning'>Save & Convert</Button>
  </Stack>;
}
