// TODO: set up i18next
export const MODEL_LABELS: Record<string, string> = {
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
  // extras
  'diffusion-stablydiffused-aesthetic-v2-6': 'Aesthetic Mix v2.6',
  'diffusion-anything': 'Anything',
  'diffusion-anything-v3': 'Anything v3',
  'diffusion-anything-v4': 'Anything v4',
  'diffusion-darkvictorian': 'Dark Victorian',
  'diffusion-dreamlike-photoreal': 'Dreamlike Photoreal',
  'diffusion-dreamlike-photoreal-v1': 'Dreamlike Photoreal 1.0',
  'diffusion-dreamlike-photoreal-v2': 'Dreamlike Photoreal 2.0',
  'diffusion-ghibli': 'Ghibli',
  'diffusion-knollingcase': 'Knollingcase',
  'diffusion-openjourney': 'OpenJourney',
  'diffusion-openjourney-v1': 'OpenJourney v1',
  'diffusion-openjourney-v2': 'OpenJourney v2',
  'diffusion-pastel-mix': 'Pastel Mix',
  'diffusion-unstable-ink-dream-v6': 'Unstable Ink Dream v6',
};

export const INVERSION_LABELS: Record<string, string> = {
  '': 'None',
  'inversion-cubex': 'Cubex',
  'inversion-birb': 'Birb Style',
  'inversion-line-art': 'Line Art',
  'inversion-minecraft': 'Minecraft Concept',
};

export const PLATFORM_LABELS: Record<string, string> = {
  amd: 'AMD GPU',
  // eslint-disable-next-line id-blacklist
  any: 'Any Platform',
  cpu: 'CPU',
  cuda: 'CUDA',
  directml: 'DirectML',
  nvidia: 'Nvidia GPU',
  rocm: 'ROCm',
};

export const SCHEDULER_LABELS: Record<string, string> = {
  'ddim': 'DDIM',
  'ddpm': 'DDPM',
  'deis-multi': 'DEIS Multistep',
  'dpm-multi': 'DPM Multistep',
  'dpm-single': 'DPM Singlestep',
  'euler': 'Euler',
  'euler-a': 'Euler Ancestral',
  'heun': 'Heun',
  'k-dpm-2-a': 'KDPM2 Ancestral',
  'k-dpm-2': 'KDPM2',
  'karras-ve': 'Karras Ve',
  'ipndm': 'iPNDM',
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
