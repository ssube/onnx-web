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
  'euler': 'Euler',
  'euler-a': 'Euler Ancestral',
  'lms-discrete': 'LMS Discrete',
  'pndm': 'PNDM',
};
