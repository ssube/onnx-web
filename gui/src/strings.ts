// TODO: set up i18next
export const MODEL_LABELS = {
  'stable-diffusion-onnx-v1-4': 'Stable Diffusion v1.4',
  'stable-diffusion-onnx-v1-5': 'Stable Diffusion v1.5',
  'stable-diffusion-onnx-v2-0': 'Stable Diffusion v2.0',
  'stable-diffusion-onnx-v2-1': 'Stable Diffusion v2.1',
};

export const PLATFORM_LABELS: Record<string, string> = {
  amd: 'AMD GPU',
  cpu: 'CPU',
  nvidia: 'Nvidia GPU',
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

export const NOISE_LABELS: Record<string, string> = {
  fill: 'Fill Edges',
  gaussian: 'Gaussian Blur',
  histogram: 'Histogram Noise',
  normal: 'Gaussian Noise',
  uniform: 'Uniform Noise',
};

export const MASK_LABELS: Record<string, string> = {
  'none': 'None',
  'gaussian-multiply': 'Gaussian Multiply',
  'gaussian-screen': 'Gaussian Screen',
};
