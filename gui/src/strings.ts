// TODO: set up i18next
export const MODEL_LABELS = {
  'stable-diffusion-onnx-v1-4': 'Stable Diffusion v1.4',
  'stable-diffusion-onnx-v1-5': 'Stable Diffusion v1.5',
  'stable-diffusion-onnx-v1-inpainting': 'SD Inpainting v1',
  'stable-diffusion-onnx-v2-0': 'Stable Diffusion v2.0',
  'stable-diffusion-onnx-v2-1': 'Stable Diffusion v2.1',
  'stable-diffusion-onnx-v2-inpainting': 'SD Inpainting v2',
  // upscaling
  'upscaling-real-esrgan-x2-plus': 'Real ESRGAN x2 Plus',
  'upscaling-real-esrgan-x4-plus': 'Real ESRGAN x4 Plus',
  'upscaling-real-esrgan-x4-v3': 'Real ESRGAN x4 v3',
  'upscaling-stable-diffusion-x4': 'Stable Diffusion x4',
  // correction
  'correction-codeformer': 'CodeFormer',
  'correction-gfpgan-v1-3': 'GFPGAN v1.3',
};

export const PLATFORM_LABELS: Record<string, string> = {
  amd: 'AMD GPU',
  cpu: 'CPU',
  cuda: 'CUDA',
  directml: 'DirectML',
  nvidia: 'Nvidia GPU',
  rocm: 'ROCm',
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
  'fill-edge': 'Fill Edges',
  'fill-mask': 'Fill Masked',
  'gaussian': 'Gaussian Blur',
  'histogram': 'Histogram Noise',
  'normal': 'Gaussian Noise',
  'uniform': 'Uniform Noise',
};

export const MASK_LABELS: Record<string, string> = {
  'none': 'None',
  'gaussian-multiply': 'Gaussian Multiply',
  'gaussian-screen': 'Gaussian Screen',
};
