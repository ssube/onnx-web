# Converting Models

This guide describes the process for converting models and additional networks to the directories used by `diffusers`
and on to the ONNX models used by `onnx-web`.

Using the `extras.json` file, you can convert SD and diffusers models to ONNX, and blend them with LoRA weights and
Textual Inversion embeddings.

## Contents

- [Converting Models](#converting-models)
  - [Contents](#contents)
  - [Conversion steps for each type of model](#conversion-steps-for-each-type-of-model)
  - [Converting diffusers models](#converting-diffusers-models)
  - [Converting SD and Dreambooth checkpoints](#converting-sd-and-dreambooth-checkpoints)
  - [Converting LoRA weights](#converting-lora-weights)
    - [Figuring out which script produced the LoRA weights](#figuring-out-which-script-produced-the-lora-weights)
    - [LoRA weights from cloneofsimo/lora](#lora-weights-from-cloneofsimolora)
    - [LoRA weights from kohya-ss/sd-scripts](#lora-weights-from-kohya-sssd-scripts)
  - [Converting Textual Inversion embeddings](#converting-textual-inversion-embeddings)
    - [Figuring out how many layers are in a Textual Inversion](#figuring-out-how-many-layers-are-in-a-textual-inversion)
  - [Optimizing diffusers models](#optimizing-diffusers-models)
    - [Converting to float16](#converting-to-float16)
    - [Optimizing with ONNX runtime](#optimizing-with-onnx-runtime)
    - [Optimizing with HuggingFace Optimum](#optimizing-with-huggingface-optimum)

## Conversion steps for each type of model

You can start from a diffusers directory, HuggingFace Hub repository, or an SD checkpoint in the form of a `.ckpt` or
`.safetensors` file:

1. LoRA weights from `kohya-ss/sd-scripts` to...
2. SD or Dreambooth checkpoint to...
3. diffusers directory or LoRA weights from `cloneofsimo/lora` to...
4. ONNX models

LoRAs and Textual inversions can be temporarily blended with an ONNX model while the server is running using prompt
tokens or permanently blended during model conversion using the `extras.json` file. LoRA and Textual Inversion models
do not need to be converted to ONNX to be used with prompt tokens.

If you are using the Auto1111 web UI or another tool, you may not need to convert the models to ONNX. In that case,
you will not have an `extras.json` file and should skip the last step.

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

You can merge one or more sets of LoRA weights into their base models using your `extras.json` file, which is directly
supported by the conversion script in `onnx-web` with no additional steps.

This is not required to use LoRA weights in the prompt, but it can save memory and enable better caching for
commonly-used model combinations.

LoRA weights produced by the `cloneofsimo/lora` repository can be converted to a diffusers directory and from there
on to ONNX, while LoRA weights produced by the `kohya-ss/sd-scripts` repository must be converted to an SD checkpoint,
which can be converted into a diffusers directory and finally ONNX models.

### Figuring out which script produced the LoRA weights

Weights exported by the two repositories are not compatible with the other and you must use the same scripts that
originally created a set of weights to merge them.

If you have a `.safetensors` file, check the metadata keys:

```python
>>> import safetensors
>>> t = safetensors.safe_open("/home/ssube/lora-weights/jack.safetensors", framework="pt")
>>> print(t.metadata())
{'ss_batch_size_per_device': '1', 'ss_bucket_info': 'null', 'ss_cache_latents': 'True', 'ss_clip_skip': '2', ...}
```

If they start with `lora_`, it's probably from the `cloneofsimo/lora` scripts. If they start with `ss_`, it's
probably from the `kohya-ss/sd-scripts` scripts.

If you get an error about missing metadata, try the other repository. For example:

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

### LoRA weights from cloneofsimo/lora

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

### LoRA weights from kohya-ss/sd-scripts

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

## Converting Textual Inversion embeddings

You can convert Textual Inversion embeddings by merging their weights and tokens into a copy of their base model,
which is directly supported by the conversion script in `onnx-web` with no additional steps.

This is not required to use LoRA weights in the prompt, but it can save memory and enable better caching for
commonly-used model combinations.

Some Textual Inversions may have more than one set of weights, which can be used and controlled separately. Some
Textual Inversions may provide their own token, but you can always use the filename to activate them in `onnx-web`.

### Figuring out how many layers are in a Textual Inversion

Textual Inversions produced by [the Stable Conceptualizer notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb)
only have a single layer, while many others have more than one.

The number of layers is shown in the server logs when the model is converted:

```none
[2023-03-08 04:54:00,234] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: found embedding for token <concept>: torch.Size([768])
[2023-03-08 04:54:01,624] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: added 1 tokens
[2023-03-08 04:54:01,814] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: saving tokenizer for Textual Inversion
[2023-03-08 04:54:01,859] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: saving text encoder for Textual Inversion
...
[2023-03-08 04:58:06,378] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: generating 74 layer tokens
[2023-03-08 04:58:06,379] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: found embedding for token ['goblin-0', 'goblin-1', 'goblin-2', 'goblin-3', 'goblin-4', 'goblin-5', 'goblin-6', 'gob
lin-7', 'goblin-8', 'goblin-9', 'goblin-10', 'goblin-11', 'goblin-12', 'goblin-13', 'goblin-14', 'goblin-15', 'goblin-16', 'goblin-17', 'goblin-18', 'goblin-19', 'goblin-20', 'goblin-21', 'goblin-22', 'goblin-23', 'goblin-24', 'goblin-25', 'goblin-26', 'goblin-27', 'goblin-28', 'goblin-29', 'goblin-30', 'goblin-31', 'goblin-32', 'goblin-33', 'goblin-34', 'goblin-35', 'goblin-36', 'goblin-37', 'goblin-38', 'goblin-39', 'goblin-40', 'goblin-41', 'goblin-42', 'goblin-43', 'goblin-44', 'goblin-45', 'goblin-46', 'goblin-47', 'goblin-48', 'goblin-49', 'goblin-50', 'goblin-51', 'goblin-52', 'goblin-53', 'goblin-54', 'goblin-55', 'goblin-56', 'goblin-57', 'goblin-58', 'goblin-59', 'goblin-60', 'goblin-61', 'goblin-62', 'goblin-63', 'goblin-64', 'goblin-65', 'goblin-66', 'goblin-67', 'goblin-68', 'goblin-69', 'goblin-70', 'goblin-71', 'goblin-72', 'goblin-73'] (*): torch.Size([74, 768])
[2023-03-08 04:58:07,685] INFO: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: added 74 tokens
[2023-03-08 04:58:07,874] DEBUG: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: embedding torch.Size([768]) vector for layer goblin-0
[2023-03-08 04:58:07,874] DEBUG: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: embedding torch.Size([768]) vector for layer goblin-1
[2023-03-08 04:58:07,874] DEBUG: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: embedding torch.Size([768]) vector for layer goblin-2
[2023-03-08 04:58:07,875] DEBUG: MainProcess MainThread onnx_web.convert.diffusion.textual_inversion: embedding torch.Size([768]) vector for layer goblin-3
```

You do not need to know how many layers a Textual Inversion has to use the base token, `goblin` or `goblin-all` in this
example, but it does allow you to control the layers individually.

## Optimizing diffusers models

The ONNX models often include redundant nodes, like division by 1, and are converted using 32-bit floating point
numbers by default. The models can be optimized to remove some of those nodes and reduce their size, both on disk and
in VRAM.

The highest levels of optimization will make the converted models platform-specific and must be done after blending
LoRAs and Textual Inversions, so you cannot select them in the prompt, but reduces memory usage by 50-75%.

### Converting to float16

The size of a model can be roughly cut in half, on disk and in memory, by converting it from float32 to float16. There
are a few different levels of conversion, which become increasingly platforms-specific.

1. Internal conversion
   - Converts graph nodes to float16 operations
   - Leaves inputs and outputs as float32
   - Initializer data can be converted to float16 or kept as float32
2. Full conversion
   - Can be done with ONNX runtime or Torch
   - Converts inputs, outputs, nodes, and initializer data to float16
   - Breaks runtime LoRA and Textual Inversion blending
   - Requires some additional data conversions at runtime, which may introduce subtle rounding errors

Using Stable Diffusion v1.5 as an example, full conversion reduces the size of the model by about half:

```none
4.0G    ./stable-diffusion-v1-5-fp32
4.0G    ./stable-diffusion-v1-5-fp32-optimized
2.6G    ./stable-diffusion-v1-5-fp16-internal
2.3G    ./stable-diffusion-v1-5-fp16-optimized
2.0G    ./stable-diffusion-v1-5-fp16-torch
```

Combined with [the other ONNX optimizations](server-admin.md#pipeline-optimizations), this can make the pipeline usable
on 4-6GB GPUs and allow much larger batch sizes on GPUs with more memory. The optimized float32 model uses somewhat
less VRAM than the original model, despite being the same size on disk.

### Optimizing with ONNX runtime

The ONNX runtime provides an optimization script for Stable Diffusion models in their git repository. You will need to
clone that repository, but you can use an existing virtual environment for `onnx-web` and should not need to install
any new packages.

```shell
> git clone https://github.com/microsoft/onnxruntime
> cd onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion
> python3 optimize_pipeline.py -i /home/ssube/onnx-web/models/stable-diffusion
```

The `optimize_pipeline.py` script should work on any [diffusers directory with ONNX models](#converting-diffusers-models),
but you will need to use the `--use_external_data_format` option if you are not using `--float16`. See the `--help` for
more details.

- https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers/models/stable_diffusion

### Optimizing with HuggingFace Optimum

- https://huggingface.co/docs/optimum/v1.7.1/en/onnxruntime/usage_guides/optimization#optimizing-a-model-with-optimum-cli
