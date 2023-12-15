import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Accordion, AccordionDetails, AccordionSummary, Alert, Button, CircularProgress, Stack } from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import _ from 'lodash';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { STALE_TIME } from '../../config.js';
import { ClientContext, OnnxState, StateContext } from '../../state/full.js';
import {
  CorrectionModel,
  DiffusionModel,
  ExtraNetwork,
  ExtrasFile,
  ExtraSource,
  NetworkModel,
  NetworkType,
  SafetensorFormat,
  UpscalingModel,
} from '../../types/model.js';
import { EditableList } from '../input/EditableList';
import { CorrectionModelInput } from '../input/model/CorrectionModel.js';
import { DiffusionModelInput } from '../input/model/DiffusionModel.js';
import { ExtraNetworkInput } from '../input/model/ExtraNetwork.js';
import { ExtraSourceInput } from '../input/model/ExtraSource.js';
import { UpscalingModelInput } from '../input/model/UpscalingModel.js';

const { useContext, useEffect } = React;
// eslint-disable-next-line @typescript-eslint/unbound-method
const { kebabCase } = _;

function mergeModelLists<T extends DiffusionModel | ExtraSource>(local: Array<T>, server: Array<T> = []) {
  const localNames = new Set(local.map((it) => it.name));

  const merged = [...local];
  for (const model of server) {
    if (localNames.has(model.name) === false) {
      merged.push(model);
    }
  }

  return merged;
}

function mergeModels(local: ExtrasFile, server: Partial<ExtrasFile>): ExtrasFile {
  const merged: ExtrasFile = {
    ...server,
    correction: mergeModelLists(local.correction, server.correction),
    diffusion: mergeModelLists(local.diffusion, server.diffusion),
    networks: mergeModelLists(local.networks, server.networks),
    sources: mergeModelLists(local.sources, server.sources),
    upscaling: mergeModelLists(local.upscaling, server.upscaling),
  };

  return merged;
}

function selectDiffusionModels(state: OnnxState): Array<DiffusionModel> {
  return state.extras.diffusion;
}

function selectCorrectionModels(state: OnnxState): Array<CorrectionModel> {
  return state.extras.correction;
}

function selectUpscalingModels(state: OnnxState): Array<UpscalingModel> {
  return state.extras.upscaling;
}

function selectExtraNetworks(state: OnnxState): Array<ExtraNetwork> {
  return state.extras.networks;
}

function selectExtraSources(state: OnnxState): Array<ExtraSource> {
  return state.extras.sources;
}

export function Models() {
  const store = mustExist(useContext(StateContext));
  const {
    setCorrectionModel,
    setDiffusionModel,
    setExtraNetwork,
    setExtraSource,
    setExtras,
    setUpscalingModel,
    removeCorrectionModel,
    removeDiffusionModel,
    removeExtraNetwork,
    removeExtraSource,
    removeUpscalingModel,
  } = useStore(store, selectActions, shallow);

  const client = mustExist(useContext(ClientContext));
  const result = useQuery(['extras'], async () => client.extras(), {
    staleTime: STALE_TIME,
  });

  const query = useQueryClient();
  const write = useMutation(writeExtras, {
    onSuccess: () => query.invalidateQueries(['extras']),
  });
  const { t } = useTranslation();

  useEffect(() => {
    if (result.status === 'success' && doesExist(result.data)) {
      setExtras(mergeModels(store.getState().extras, result.data));
    }
  }, [result.status]);

  if (result.status === 'error') {
    return <Stack spacing={2} direction='row' sx={{ alignItems: 'center' }}>
      <Alert severity='error'>Error</Alert>
    </Stack>;
  }

  if (result.status === 'loading') {
    return <Stack spacing={2} direction='row' sx={{ alignItems: 'center' }}>
      <CircularProgress />
    </Stack>;
  }

  async function writeExtras() {
    const resp = await client.writeExtras(store.getState().extras);
    // TODO: do something with resp
  }

  return <Stack spacing={2}>
    <Accordion>
      <AccordionSummary>
        {t('modelType.diffusion', { count: 10 })}
      </AccordionSummary>
      <AccordionDetails>
        <EditableList<DiffusionModel>
          selector={selectDiffusionModels}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: `diffusion-${kebabCase(l)}`,
            source: s,
          })}
          removeItem={(m) => removeDiffusionModel(m)}
          renderItem={DiffusionModelInput}
          setItem={(model) => setDiffusionModel(model)}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        {t('modelType.correction', { count: 10 })}
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          selector={selectCorrectionModels}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: `correction-${kebabCase(l)}`,
            source: s,
          })}
          removeItem={(m) => removeCorrectionModel(m)}
          renderItem={CorrectionModelInput}
          setItem={(model) => setCorrectionModel(model)}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        {t('modelType.upscaling', { count: 10 })}
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          selector={selectUpscalingModels}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: `upscaling-${kebabCase(l)}`,
            scale: 4,
            source: s,
          })}
          removeItem={(m) => removeUpscalingModel(m)}
          renderItem={UpscalingModelInput}
          setItem={(model) => setUpscalingModel(model)}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        {t('modelType.network', { count: 10 })}
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          selector={selectExtraNetworks}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            model: 'embeddings' as NetworkModel,
            name: kebabCase(l),
            source: s,
            type: 'inversion' as NetworkType,
          })}
          removeItem={(m) => removeExtraNetwork(m)}
          renderItem={ExtraNetworkInput}
          setItem={(model) => setExtraNetwork(model)}
        />
      </AccordionDetails>
    </Accordion>
    <Accordion>
      <AccordionSummary>
        {t('modelType.source', { count: 10 })}
      </AccordionSummary>
      <AccordionDetails>
        <EditableList
          selector={selectExtraSources}
          newItem={(l, s) => ({
            format: 'safetensors' as SafetensorFormat,
            label: l,
            name: kebabCase(l),
            source: s,
          })}
          removeItem={(m) => removeExtraSource(m)}
          renderItem={ExtraSourceInput}
          setItem={(model) => setExtraSource(model)}
        />
      </AccordionDetails>
    </Accordion>
    <Button color='warning' onClick={() => write.mutate()}>{t('convert')}</Button>
  </Stack>;
}

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setExtras: state.setExtras,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setCorrectionModel: state.setCorrectionModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setDiffusionModel: state.setDiffusionModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setExtraNetwork: state.setExtraNetwork,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setExtraSource: state.setExtraSource,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscalingModel: state.setUpscalingModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeCorrectionModel: state.removeCorrectionModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeDiffusionModel: state.removeDiffusionModel,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeExtraNetwork: state.removeExtraNetwork,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeExtraSource: state.removeExtraSource,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    removeUpscalingModel: state.removeUpscalingModel,
  };
}
