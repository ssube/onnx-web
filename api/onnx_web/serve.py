from diffusers import (
    # schedulers
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KarrasVeScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    # onnx
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    # types
    DiffusionPipeline,
)
from flask import Flask, jsonify, request, send_from_directory, url_for
from hashlib import sha256
from io import BytesIO
from PIL import Image
from struct import pack
from os import environ, makedirs, path, scandir
from typing import Any, Tuple, Union

import json
import numpy as np

# defaults
default_model = 'stable-diffusion-onnx-v1-5'
default_platform = 'amd'
default_scheduler = 'euler-a'
default_prompt = "a photo of an astronaut eating a hamburger"
default_cfg = 8
default_steps = 20
default_height = 512
default_width = 512
default_strength = 0.5

max_cfg = 30
max_steps = 150
max_height = 512
max_width = 512

# paths
model_path = environ.get('ONNX_WEB_MODEL_PATH', '../models')
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', '../outputs')
params_path = environ.get('ONNX_WEB_PARAMS_PATH', './params.json')


# pipeline caching
available_models = []
config_params = {}
last_pipeline_instance = None
last_pipeline_options = (None, None, None)
last_pipeline_scheduler = None

# pipeline params
platform_providers = {
    'amd': 'DmlExecutionProvider',
    'cpu': 'CPUExecutionProvider',
    'nvidia': 'CUDAExecutionProvider',
}
pipeline_schedulers = {
    'ddim': DDIMScheduler,
    'ddpm': DDPMScheduler,
    'dpm-multi': DPMSolverMultistepScheduler,
    'dpm-single': DPMSolverSinglestepScheduler,
    'euler': EulerDiscreteScheduler,
    'euler-a': EulerAncestralDiscreteScheduler,
    'heun': HeunDiscreteScheduler,
    'k-dpm-2-a': KDPM2AncestralDiscreteScheduler,
    'k-dpm-2': KDPM2DiscreteScheduler,
    'karras-ve': KarrasVeScheduler,
    'lms-discrete': LMSDiscreteScheduler,
    'pndm': PNDMScheduler,
}


def get_and_clamp_float(args, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key: str, values: dict[str, Any], default: Any):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


def get_model_path(model: str):
    return safer_join(model_path, model)


# from https://www.travelneil.com/stable-diffusion-updates.html
def get_latents_from_seed(seed: int, width: int, height: int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def load_pipeline(pipeline: DiffusionPipeline, model: str, provider: str, scheduler):
    global last_pipeline_instance
    global last_pipeline_scheduler
    global last_pipeline_options

    options = (pipeline, model, provider)
    if last_pipeline_instance != None and last_pipeline_options == options:
        print('reusing existing pipeline')
        pipe = last_pipeline_instance
    else:
        print('loading different pipeline')
        pipe = pipeline.from_pretrained(
            model,
            provider=provider,
            safety_checker=None,
            scheduler=scheduler.from_pretrained(model, subfolder='scheduler')
        )
        last_pipeline_instance = pipe
        last_pipeline_options = options
        last_pipeline_scheduler = scheduler

    if last_pipeline_scheduler != scheduler:
        print('changing pipeline scheduler')
        pipe.scheduler = scheduler.from_pretrained(
            model, subfolder='scheduler')
        last_pipeline_scheduler = scheduler

    return pipe


def json_with_cors(data, origin='*'):
    """Build a JSON response with CORS headers allowing `origin`"""
    res = jsonify(data)
    res.access_control_allow_origin = origin
    return res


def make_output_path(type: str, seed: int, params: Tuple[Union[str, int, float]]):
    sha = sha256()
    sha.update(type.encode('utf-8'))
    for param in params:
        if isinstance(param, str):
            sha.update(param.encode('utf-8'))
        elif isinstance(param, int):
            sha.update(bytearray(pack('!I', param)))
        elif isinstance(param, float):
            sha.update(bytearray(pack('!f', param)))
        else:
            print('cannot hash param: %s, %s' % (param, type(param)))

    output_file = '%s_%s_%s.png' % (type, seed, sha.hexdigest())
    output_full = safer_join(output_path, output_file)

    return (output_file, output_full)


def safer_join(base, tail):
    safer_path = path.relpath(path.normpath(path.join('/', tail)), '/')
    return path.join(base, safer_path)


def url_from_rule(rule):
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)

# setup


def check_paths():
    if not path.exists(model_path):
        raise RuntimeError('model path must exist')

    if not path.exists(output_path):
        makedirs(output_path)


def load_models():
    global available_models
    available_models = [f.name for f in scandir(model_path) if f.is_dir()]


def load_params():
    global config_params
    with open(params_path) as f:
        config_params = json.load(f)


check_paths()
load_models()
load_params()
app = Flask(__name__)

# routes


@app.route('/')
def index():
    return {
        'name': 'onnx-web',
        'routes': [{
            'path': url_from_rule(rule),
            'methods': list(rule.methods).sort()
        } for rule in app.url_map.iter_rules()]
    }


@app.route('/settings/models')
def list_models():
    return json_with_cors(available_models)


@app.route('/settings/params')
def list_params():
    return json_with_cors(config_params)


@app.route('/settings/platforms')
def list_platforms():
    return json_with_cors(list(platform_providers.keys()))


@app.route('/settings/schedulers')
def list_schedulers():
    return json_with_cors(list(pipeline_schedulers.keys()))


def pipeline_from_request(pipeline: DiffusionPipeline):
    user = request.remote_addr

    # pipeline stuff
    model = get_model_path(request.args.get('model', default_model))
    provider = get_from_map(request.args, 'platform',
                            platform_providers, default_platform)
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, default_scheduler)

    # image params
    prompt = request.args.get('prompt', default_prompt)
    negative_prompt = request.args.get('negative', None)

    cfg = get_and_clamp_int(request.args, 'cfg', default_cfg, config_params.get('cfg').get('max'), 0)
    steps = get_and_clamp_int(request.args, 'steps', default_steps, config_params.get('steps').get('max'))
    height = get_and_clamp_int(
        request.args, 'height', default_height, config_params.get('height').get('max'))
    width = get_and_clamp_int(request.args, 'width', default_width, config_params.get('width').get('max'))

    seed = int(request.args.get('seed', -1))
    if seed == -1:
        seed = np.random.randint(np.iinfo(np.int32).max)

    print("request from %s: %s rounds of %s using %s on %s, %sx%s, %s, %s - %s" %
          (user, steps, scheduler.__name__, model, provider, width, height, cfg, seed, prompt))

    pipe = load_pipeline(pipeline, model, provider, scheduler)
    return (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height, width, seed, pipe)


@app.route('/img2img', methods=['POST'])
def img2img():
    input_file = request.files.get('source')
    input_image = Image.open(BytesIO(input_file.read())).convert('RGB')
    input_image.thumbnail((default_width, default_height))

    strength = get_and_clamp_float(request.args, 'strength', 0.5, 1.0)

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed, pipe) = pipeline_from_request(OnnxStableDiffusionImg2ImgPipeline)

    rng = np.random.RandomState(seed)
    image = pipe(
        prompt,
        generator=rng,
        guidance_scale=cfg,
        image=input_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        strength=strength,
    ).images[0]

    (output_file, output_full) = make_output_path('img2img', seed,
                                                  (prompt, cfg, negative_prompt, steps, strength, height, width))
    print("img2img output: %s" % output_full)
    image.save(output_full)

    return json_with_cors({
        'output': output_file,
        'params': {
            'model': model,
            'provider': provider,
            'scheduler': scheduler.__name__,
            'seed': seed,
            'prompt': prompt,
            'cfg': cfg,
            'negativePrompt': negative_prompt,
            'steps': steps,
            'height': default_height,
            'width': default_width,
        }
    })


@app.route('/txt2img', methods=['POST'])
def txt2img():
    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed, pipe) = pipeline_from_request(OnnxStableDiffusionPipeline)

    latents = get_latents_from_seed(seed, width, height)
    rng = np.random.RandomState(seed)

    image = pipe(
        prompt,
        height,
        width,
        generator=rng,
        guidance_scale=cfg,
        latents=latents,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
    ).images[0]

    (output_file, output_full) = make_output_path('txt2img',
                                                  seed, (prompt, cfg, negative_prompt, steps, height, width))
    print("txt2img output: %s" % output_full)
    image.save(output_full)

    return json_with_cors({
        'output': output_file,
        'params': {
            'model': model,
            'provider': provider,
            'scheduler': scheduler.__name__,
            'seed': seed,
            'prompt': prompt,
            'cfg': cfg,
            'negativePrompt': negative_prompt,
            'steps': steps,
            'height': height,
            'width': width,
        }
    })


@app.route('/inpaint', methods=['POST'])
def inpaint():
    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')
    source_image.thumbnail((default_width, default_height))

    mask_file = request.files.get('mask')
    mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')
    mask_image.thumbnail((default_width, default_height))

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed, pipe) = pipeline_from_request(OnnxStableDiffusionInpaintPipeline)

    latents = get_latents_from_seed(seed, width, height)
    rng = np.random.RandomState(seed)

    image = pipe(
        prompt,
        generator=rng,
        guidance_scale=cfg,
        height=height,
        image=source_image,
        latents=latents,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
    ).images[0]

    (output_file, output_full) = make_output_path(
        'inpaint', (prompt, cfg, steps, height, width, seed))
    print("inpaint output: %s" % output_full)
    image.save(output_full)

    return json_with_cors({
        'output': output_file,
        'params': {
            'model': model,
            'provider': provider,
            'scheduler': scheduler.__name__,
            'seed': seed,
            'prompt': prompt,
            'cfg': cfg,
            'negativePrompt': negative_prompt,
            'steps': steps,
            'height': default_height,
            'width': default_width,
        }
    })


@app.route('/output/<path:filename>')
def output(filename: str):
    return send_from_directory(path.join('..', output_path), filename, as_attachment=False)
