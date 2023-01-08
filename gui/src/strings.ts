// TODO: set up i18next
export const MODEL_LABELS = {
  'stable-diffusion-onnx-v1-5': 'Stable Diffusion v1.5',
};

export const PLATFORM_LABELS: Record<string, string> = {
  amd: 'AMD GPU',
  cpu: 'CPU',
};

export const SCHEDULER_LABELS: Record<string, string> = {
  'ddim': 'DDIM',
  'ddpm': 'DDPM',
  'dpm-multi': 'DPM Multistep',
  'dpm-single': 'DPM Singlestep',
  'euler': 'Euler',
  'euler-a': 'Euler Ancestral',
  'heun': 'Heun',
  'k-dpm-2-a': 'KDPM2 Ancestral',
  'k-dpm-2': 'KDPM2',
  'karras-ve': 'Karras Ve',
  'lms-discrete': 'LMS',
  'pndm': 'PNDM',
};
