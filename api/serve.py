from diffusers import OnnxStableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from flask import Flask, jsonify, request, send_from_directory
from stringcase import spinalcase
from os import environ, path, makedirs
import numpy as np

# defaults
default_prompt = "a photo of an astronaut eating a hamburger"
default_cfg = 8
default_steps = 20
default_height = 512
default_width = 512

max_cfg = 30
max_steps = 150
max_height = 512
max_width = 512

# paths
model_path = environ.get('ONNX_WEB_MODEL_PATH', "../models/stable-diffusion-onnx-v1-5")
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', "../outputs")

# platforms
platform_providers = {
    'amd': 'DmlExecutionProvider',
    'cpu': 'CPUExecutionProvider',
}

# schedulers
pipeline_schedulers = {
    'ddim': DDIMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'ddpm': DDPMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'dpm-multi': DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'euler': EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'euler-a': EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'lms-discrete': LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
    'pndm': PNDMScheduler.from_pretrained(model_path, subfolder="scheduler"),
}


def get_and_clamp(args, key, default_value, max_value, min_value=1):
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key, values, default):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


# TODO: credit this function
def get_latents_from_seed(seed: int, width: int, height: int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


# setup
if not path.exists(model_path):
    raise RuntimeError('model path must exist')

if not path.exists(output_path):
    makedirs(output_path)

app = Flask(__name__)

# routes


@app.route('/')
def hello():
    return 'Hello, %s' % (__name__)


@app.route('/txt2img')
def txt2img():
    user = request.remote_addr

    prompt = request.args.get('prompt', default_prompt)
    provider = get_from_map(request.args, 'provider', platform_providers, 'amd')
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, 'euler-a')
    cfg = get_and_clamp(request.args, 'cfg', default_cfg, max_cfg, 0)
    steps = get_and_clamp(request.args, 'steps', default_steps, max_steps)
    height = get_and_clamp(request.args, 'height', default_height, max_height)
    width = get_and_clamp(request.args, 'width', default_width, max_width)

    seed = int(request.args.get('seed', -1))
    if seed == -1:
        seed = np.random.randint(np.iinfo(np.int32).max)

    latents = get_latents_from_seed(seed, width, height)

    print("txt2img from %s: %s/%s, %sx%s, %s, %s" %
          (user, cfg, steps, width, height, seed, prompt))

    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        model_path,
        provider=provider,
        safety_checker=None,
        scheduler=scheduler
    )
    image = pipe(
        prompt,
        height,
        width,
        num_inference_steps=steps,
        guidance_scale=cfg,
        latents=latents
    ).images[0]

    output_file = "txt2img_%s_%s.png" % (seed, spinalcase(prompt[0:64]))
    output_full = '%s/%s' % (output_path, output_file)
    print("txt2img output: %s" % output_full)
    image.save(output_full)

    res = jsonify({
        'output': output_file,
        'params': {
            'cfg': cfg,
            'steps': steps,
            'height': height,
            'width': width,
            'prompt': prompt,
            'seed': seed
        }
    })
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res


@app.route('/output/<path:filename>')
def output(filename):
    return send_from_directory(output_path, filename, as_attachment=False)
