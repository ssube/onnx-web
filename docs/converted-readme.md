# TODO MODEL TITLE

This is a copy of TODO MODEL TITLE, TODO CIVITAI LINK, converted to the ONNX format for use with tools that use ONNX models, such
as https://github.com/ssube/onnx-web. If you have questions about using this model, please see
https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#pre-converted-models.

FP16 WARNING: This model has been converted to FP16 format and will not run correctly on the CPU platform. If you are
using the CPU platform, please use the FP32 model instead.

As a derivative of TODO MODEL TITLE, the model files that came with this README are licensed under the terms of the TODO LICENSE. A copy
of the license was included in the archive. Please make sure to read and follow the terms before you use this model or
redistribute these files.

If you are the author of this model and have questions about ONNX models or would like to have this model removed from
distribution or moved to another site, please contact ssube on https://github.com/ssube/onnx-web/issues or
https://discord.gg/7CdQmutGuw.

## Adding models

Extract the entire ZIP archive into the models folder of your onnx-web installation and restart the server or click the
Restart Workers button in the web UI and then refresh the page.

Please see https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#adding-your-own-models for more details.

## Folder structure

- cnet
  - UNet model with ControlNet layers
  - remove for SDXL
- LICENSE.txt
  - the original model's license
- model_index.json
  - links the other models together
- README.txt
  - this readme
- scheduler
  - scheduler config
- text_encoder
  - text encoder model
- text_encoder2
  - second text encoder model for SDXL
  - remove for SD v1.5
- tokenizer
  - tokenizer config
- tokenizer2
  - second tokenizer config for SDXL
  - remove for SD v1.5
- unet
  - UNet model
- vae_decoder
  - VAE decoder model
- vae_encoder
  - VAE encoder model
