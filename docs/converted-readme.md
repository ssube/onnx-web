# MODEL TITLE

This is a copy of MODEL TITLE converted to the ONNX format for use with tools that use ONNX models, such as
https://github.com/ssube/onnx-web. If you have questions about using this model, please see
https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#pre-converted-models.

As a derivative of MODEL TITLE, the model files that came with this README are licensed under the terms of TODO. A copy
of the license was included in the archive. Please make sure to read and follow the terms before you use this model or
redistribute these files.

If you are the author of this model and have questions about ONNX models or would like to have this model removed from
distribution or moved to another site, please contact ssube on https://github.com/ssube/onnx-web/issues or
https://discord.gg/7CdQmutGuw.

## Folder structure

- cnet
  - UNet model with ControlNet layers
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
- tokenizer
  - tokenizer config
- unet
  - UNet model
- vae_decoder
  - VAE decoder model
- vae_encoder
  - VAE encoder model

## Adding models

Extract the entire ZIP archive into the models folder of your onnx-web installation and restart the server or click the
Restart Workers button in the web UI and then refresh the page.

Please see https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#adding-your-own-models for more details.
