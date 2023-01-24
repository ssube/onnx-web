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
  - 11.6
  - 11.7
- ROCm
  - 5.2
  - 5.4 seems like it might work

## Container/Platform Acceleration

| Runtime | CUDA | DirectML | ROCm     | CPU |
| ------- | ---- | -------- | -------- | --- |
| docker  | yes  | no, 1    | maybe, 2 | yes |
| podman  | 3    | no, 1    | maybe, 2 | yes |

### Container Notes

1. no package available: https://github.com/ssube/onnx-web/issues/63
2. should work but testing failed: https://github.com/ssube/onnx-web/issues/10
3. should work, not tested: https://gist.github.com/bernardomig/315534407585d5912f5616c35c7fe374

## Model/Platform Acceleration

| Model            | CUDA | DirectML | ROCm  | CPU |
| ---------------- | ---- | -------- | ----- | --- |
| Stable Diffusion | yes  | yes      | yes   | yes |
| Real ESRGAN      | yes  | yes      | no, 1 | yes |
| GFPGAN           | 2    | 2        | 2     | yes |

### Model Notes

1. Real ESRGAN running on ROCm crashes with an error:

   ```none
      File "/home/ssube/onnx-web/api/onnx_web/upscale.py", line 67, in __call__
        output = self.session.run([output_name], {
      File "/home/ssube/onnx-web/api/onnx_env/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 200, in run
        return self._sess.run(output_names, input_feed, run_options)
    onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running FusedConv node. Name:'/body/body.0/rdb1/conv1/Conv' Status Message: MIOPEN failure 1: miopenStatusNotInitialized ; GPU=0 ; hostname=ssube-notwin ; expr=status_;
    ```

2. GFPGAN seems to always be running in CPU mode
