# Compatibility

## Contents

- [Compatibility](#compatibility)
  - [Contents](#contents)
  - [Driver Versions](#driver-versions)
  - [Container/Platform Acceleration](#containerplatform-acceleration)
    - [Container Notes](#container-notes)
  - [Model/Platform Acceleration](#modelplatform-acceleration)
    - [Model Notes](#model-notes)

## Driver Versions

- CUDA
  - 11.8
  - 12.0
- ROCm
  - 5.2
  - 5.4

## Container/Platform Acceleration

| Runtime | CUDA | DirectML | ROCm     | CPU |
| ------- | ---- | -------- | -------- | --- |
| docker  | yes  | no, 1    | maybe, 2 | yes |
| podman  | 3    | no, 1    | maybe, 2 | yes |

### Container Notes

1. [no package available: #63](https://github.com/ssube/onnx-web/issues/63)
2. [should work but testing failed: #10](https://github.com/ssube/onnx-web/issues/10)
3. [should work, not tested: gist instructions](https://gist.github.com/bernardomig/315534407585d5912f5616c35c7fe374)

## Model/Platform Acceleration

| Model            | ONNX | CUDA | DirectML | ROCm  | CPU |
| ---------------- | ---- | ---- | -------- | ----- | --- |
| Stable Diffusion | yes  | yes  | yes      | yes   | yes |
| - txt2img        | yes  | ^    | ^        | ^     | ^   |
| - img2img        | yes  | ^    | ^        | ^     | ^   |
| - inpaint        | yes  | ^    | ^        | ^     | ^   |
| - upscale        | yes  | ^    | ^        | ^     | ^   |
| Real ESRGAN      | yes  | yes  | no, 1    | no, 1 | yes |
| - x2/x4 plus     | yes  | ^    | ^        | ^     | ^   |
| - v3             | yes  | ^    | ^        | ^     | ^   |
| GFPGAN           | no   | -    | -        | -     | -   |
| CodeFormer       | no   | -    | -        | -     | -   |

### Model Notes

1. Real ESRGAN running on DirectML or ROCm falls back to the CPU provider with an unspecified error
