# Converting Models

This guide describes the process for converting models from various formats, including Dreambooth and LoRA, to
the checkpoints used by `diffusers` and on to the ONNX models used by `onnx-web`.

## Contents

- [Converting Models](#converting-models)
  - [Contents](#contents)
  - [Conversion steps for each type of model](#conversion-steps-for-each-type-of-model)
  - [Converting diffusers models](#converting-diffusers-models)
  - [Converting SD and Dreambooth checkpoints](#converting-sd-and-dreambooth-checkpoints)
  - [Converting LoRA weights](#converting-lora-weights)
    - [Figuring out which script produced the LoRA weights](#figuring-out-which-script-produced-the-lora-weights)
    - [LoRA models from cloneofsimo/lora](#lora-models-from-cloneofsimolora)
    - [LoRA models from kohya-ss/sd-scripts](#lora-models-from-kohya-sssd-scripts)

## Conversion steps for each type of model

You can start from a diffusers directory, HuggingFace Hub repository, or an SD checkpoint in the form of a `.ckpt` or
`.safetensors` file:

1. LoRA weights from `kohya-ss/sd-scripts` to...
2. SD or Dreambooth checkpoint to...
3. diffusers or LoRA weights from `cloneofsimo/lora` to...
4. ONNX models

One disadvantage of using ONNX is that LoRA weights must be merged with the base model before being converted,
so the final output is roughly the size of the base model. Hopefully this can be reduced in the future.

If you are using the Auto1111 web UI or another tool, you may not need to convert the models to ONNX. In that case,
you will not have an `extras.json` file and should skip step 4.

## Converting diffusers models

This is the simplest case and is supported by the conversion script in `onnx-web` with no additional steps. You can
also use the script from the `diffusers` library.

Add an entry to your `extras.json` file for each model, using the name of the HuggingFace hub repository or a local
path:

```json
    {
      "name": "diffusion-knollingcase",
      "source": "Aybeeceedee/knollingcase"
    },
    {
      "name": "diffusion-openjourney",
      "source": "prompthero/openjourney"
    },
```

To convert the diffusers model using the `diffusers` script:

```shell
> python3 convert_stable_diffusion_checkpoint_to_onnx.py \
  --model_path="runwayml/stable-diffusion-v1-5" \
  --output_path="~/onnx-web/models/stable-diffusion-onnx-v1-5"
```

Based on docs and code in:

- https://github.com/azuritecoin/OnnxDiffusersUI#download-model-and-convert-to-onnx
- https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py

## Converting SD and Dreambooth checkpoints

This works for most of the original SD checkpoints and many Dreambooth models, like those found on
[Civitai](https://civitai.com), and is supported by the conversion script in `onnx-web` with no additional steps.
You can also use the script from the `diffusers` library.

Add an entry to your `extras.json` file for each model:, :

```json
    {
      "name": "diffusion-stablydiffused-aesthetic-v2-6",
      "source": "civitai://6266?type=Pruned%20Model&format=SafeTensor",
      "format": "safetensors"
    },
    {
      "name": "diffusion-unstable-ink-dream-v6",
      "source": "civitai://5796",
      "format": "safetensors"
    },
```

For the source, you can use the name of the HuggingFace hub repository,
[the model's download ID from Civitai](./user-guide.md#downloading-models-from-civitai) (which may not match the
display ID), or an HTTPS URL. Make sure to set the `format` to match the model that you downloaded, usually
`safetensors`. You do not need to download the file ahead of time, but if you have, you can also use a local path.

To convert an SD checkpoint using the `diffusers` script:

```shell
> python3 convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path="~/onnx-web/models/.cache/sd-v1-4.ckpt" \
    --dump_path="~/onnx-web/models/stable-diffusion-onnx-v1-5"
```

Based on docs and code in:

- https://github.com/d8ahazard/sd_dreambooth_extension
- https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py

## Converting LoRA weights

You can merge one or more sets of LoRA weights into their base models, and then use your `extras.json` file to
convert them into usable ONNX models.

LoRA weights produced by the `cloneofsimo/lora` repository can be converted to a diffusers directory and from there
on to ONNX, while LoRA weights produced by the `kohya-ss/sd-scripts` repository must be converted to an SD checkpoint,
which can be converted into a diffusers directory and finally ONNX models.

### Figuring out which script produced the LoRA weights

Weights exported by the two repositories are not compatible with the other and you must use the same scripts that
originally created a set of weights to merge them.

Try the other repository if you get an error about missing metadata, for example:

```none
  warnings.warn(
Traceback (most recent call last):
  File "/home/ssube/lora/venv/bin/lora_add", line 33, in <module>
    sys.exit(load_entry_point('lora-diffusion', 'console_scripts', 'lora_add')())
  File "/home/ssube/lora/lora_diffusion/cli_lora_add.py", line 201, in main
    fire.Fire(add)
  File "/home/ssube/lora/venv/lib/python3.10/site-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/home/ssube/lora/venv/lib/python3.10/site-packages/fire/core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/home/ssube/lora/venv/lib/python3.10/site-packages/fire/core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/home/ssube/lora/lora_diffusion/cli_lora_add.py", line 133, in add
    patch_pipe(loaded_pipeline, path_2)
  File "/home/ssube/lora/lora_diffusion/lora.py", line 1012, in patch_pipe
    monkeypatch_or_replace_safeloras(pipe, safeloras)
  File "/home/ssube/lora/lora_diffusion/lora.py", line 800, in monkeypatch_or_replace_safeloras
    loras = parse_safeloras(safeloras)
  File "/home/ssube/lora/lora_diffusion/lora.py", line 565, in parse_safeloras
    raise ValueError(
ValueError: Tensor lora_te_text_model_encoder_layers_0_mlp_fc1.alpha has no metadata - is this a Lora safetensor?
```

See https://github.com/cloneofsimo/lora/issues/191 for more information.

### LoRA models from cloneofsimo/lora

Download the `lora` repo and create a virtual environment for it:

```shell
> git clone https://github.com/cloneofsimo/lora.git
> python3 -m venv venv
> source venv/bin/activate
> pip3 install -r requirements.txt
> pip3 install accelerate
```

Download the base model and LoRA weights that you want to merge first, or provide the names of HuggingFace hub repos
when you run the `lora_add` command:

```shell
> python3 -m lora_diffusion.cli_lora_add \
    runwayml/stable-diffusion-v1-5 \
    sayakpaul/sd-model-finetuned-lora-t4 \
    ~/onnx-web/models/.cache/diffusion-sd-v1-5-pokemon \
    0.8 \
    --mode upl
```

The output is a diffusers directory (step 3) and can be converted to ONNX by adding an entry to your `extras.json`
file that matches the output path:

```json
    {
      "name": "diffusion-sd-v1-5-pokemon",
      "source": ".cache/diffusion-sd-v1-5-pokemon"
    },
```

Based on docs in:

- https://github.com/cloneofsimo/lora#merging-full-model-with-lora

### LoRA models from kohya-ss/sd-scripts

Download the `sd-scripts` repo and create a virtual environment for it:

```shell
> git clone https://github.com/kohya-ss/sd-scripts.git
> python3 -m venv venv
> source venv/bin/activate
> pip3 install -r requirements.txt
> pip3 install torch torchvision
```

Download the base model and LoRA weights that you want to merge, then run the `merge_lora.py` script:

```shell
> python networks/merge_lora.py \
    --sd_model ~/onnx-web/models/.cache/v1-5-pruned-emaonly.safetensors \
    --save_to ~/onnx-web/models/.cache/v1-5-elldreths-vivid-mix.safetensors \
    --models ~/lora-weights/elldreths-vivid-mix.safetensors \
    --ratios 1.0
```

The output is an SD checkpoint (step 2) and can be converted to ONNX by adding an entry to your `extras.json` file
that matches the `--save_to` path:

```json
    {
      "name": "diffusion-lora-elldreths-vivid-mix",
      "source": "../models/.cache/v1-5-elldreths-vivid-mix.safetensors",
      "format": "safetensors"
    },
```

Make sure to set the `format` key and that it matches the format you used to export the merged model, usually
`.safetensors`.

Based on docs in:

- https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E3%83%9E%E3%83%BC%E3%82%B8%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
