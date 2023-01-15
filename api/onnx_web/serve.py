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
from flask_cors import CORS
from flask_executor import Executor
from hashlib import sha256
from io import BytesIO
from PIL import Image
from struct import pack
from os import environ, makedirs, path, scandir
from typing import Any, Dict, Tuple, Union

from .image import (
    expand_image,
    # mask filters
    mask_filter_gaussian_multiply,
    mask_filter_gaussian_screen,
    mask_filter_none,
    # noise sources
    noise_source_fill_edge,
    noise_source_fill_mask,
    noise_source_gaussian,
    noise_source_histogram,
    noise_source_normal,
    noise_source_uniform,
)

import json
import numpy as np
import time

# paths
bundle_path = environ.get('ONNX_WEB_BUNDLE_PATH',
                          path.join('..', 'gui', 'out'))
model_path = environ.get('ONNX_WEB_MODEL_PATH', path.join('..', 'models'))
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', path.join('..', 'outputs'))
params_path = environ.get('ONNX_WEB_PARAMS_PATH', 'params.json')

# options
cors_origin = environ.get('ONNX_WEB_CORS_ORIGIN', '*').split(',')
num_workers = int(environ.get('ONNX_WEB_NUM_WORKERS', 1))

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
noise_sources = {
    'fill-edge': noise_source_fill_edge,
    'fill-mask': noise_source_fill_mask,
    'gaussian': noise_source_gaussian,
    'histogram': noise_source_histogram,
    'normal': noise_source_normal,
    'uniform': noise_source_uniform,
}
mask_filters = {
    'none': mask_filter_none,
    'gaussian-multiply': mask_filter_gaussian_multiply,
    'gaussian-screen': mask_filter_gaussian_screen,
}


def get_and_clamp_float(args, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key: str, values: Dict[str, Any], default: Any):
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


def serve_bundle_file(filename='index.html'):
    return send_from_directory(path.join('..', bundle_path), filename)


def make_output_path(mode: str, seed: int, params: Tuple[Union[str, int, float]]):
    now = int(time.time())
    sha = sha256()
    sha.update(mode.encode('utf-8'))

    for param in params:
        if param is None:
            continue
        elif isinstance(param, float):
            sha.update(bytearray(pack('!f', param)))
        elif isinstance(param, int):
            sha.update(bytearray(pack('!I', param)))
        elif isinstance(param, str):
            sha.update(param.encode('utf-8'))
        else:
            print('cannot hash param: %s, %s' % (param, type(param)))

    output_file = '%s_%s_%s_%s.png' % (mode, seed, sha.hexdigest(), now)
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


def pipeline_from_request():
    user = request.remote_addr

    # pipeline stuff
    model = get_model_path(request.args.get(
        'model', config_params.get('model').get('default')))
    provider = get_from_map(request.args, 'platform',
                            platform_providers, config_params.get('platform').get('default'))
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, config_params.get('scheduler').get('default'))

    # image params
    prompt = request.args.get(
        'prompt', config_params.get('prompt').get('default'))
    negative_prompt = request.args.get('negativePrompt', None)

    if negative_prompt is not None and negative_prompt.strip() == '':
        negative_prompt = None

    cfg = get_and_clamp_float(
        request.args, 'cfg',
        config_params.get('cfg').get('default'),
        config_params.get('cfg').get('max'),
        config_params.get('cfg').get('min'))
    steps = get_and_clamp_int(
        request.args, 'steps',
        config_params.get('steps').get('default'),
        config_params.get('steps').get('max'),
        config_params.get('steps').get('min'))
    height = get_and_clamp_int(
        request.args, 'height',
        config_params.get('height').get('default'),
        config_params.get('height').get('max'),
        config_params.get('height').get('min'))
    width = get_and_clamp_int(
        request.args, 'width',
        config_params.get('width').get('default'),
        config_params.get('width').get('max'),
        config_params.get('width').get('min'))

    seed = int(request.args.get('seed', -1))
    if seed == -1:
        seed = np.random.randint(np.iinfo(np.int32).max)

    print("request from %s: %s rounds of %s using %s on %s, %sx%s, %s, %s - %s" %
          (user, steps, scheduler.__name__, model, provider, width, height, cfg, seed, prompt))

    return (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height, width, seed)


def run_txt2img_pipeline(model, provider, scheduler, prompt, negative_prompt, cfg, steps, seed, output, height, width):
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         model, provider, scheduler)

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
    image.save(output)

    print('saved txt2img output: %s' % (output))


def run_img2img_pipeline(model, provider, scheduler, prompt, negative_prompt, cfg, steps, seed, output, strength, input_image):
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         model, provider, scheduler)

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
    image.save(output)

    print('saved img2img output: %s' % (output))


def run_inpaint_pipeline(
    model: str,
    provider: str,
    scheduler: Any,
    prompt: str,
    negative_prompt: Union[str, None],
    cfg: float,
    steps: int,
    seed: int,
    output: str,
    height: int,
    width: int,
    source_image: Image,
    mask_image: Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
    noise_source: Any,
    mask_filter: Any
):
    pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                         model, provider, scheduler)

    latents = get_latents_from_seed(seed, width, height)
    rng = np.random.RandomState(seed)

    print('applying mask filter and generating noise source')
    source_image, mask_image, noise_image, _full_dims = expand_image(
        source_image,
        mask_image,
        (left, right, top, bottom),
        noise_source=noise_source,
        mask_filter=mask_filter)

    if environ.get('DEBUG') is not None:
        source_image.save(safer_join(output_path, 'last-source.png'))
        mask_image.save(safer_join(output_path, 'last-mask.png'))
        noise_image.save(safer_join(output_path, 'last-noise.png'))

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

    image.save(output)

    print('saved inpaint output: %s' % (output))


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
app.config['EXECUTOR_MAX_WORKERS'] = num_workers
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True

CORS(app, origins=cors_origin)
executor = Executor(app)

# routes


@app.route('/')
def index():
    return serve_bundle_file()


@app.route('/<path:filename>')
def index_path(filename):
    return serve_bundle_file(filename)


@app.route('/api')
def introspect():
    return {
        'name': 'onnx-web',
        'routes': [{
            'path': url_from_rule(rule),
            'methods': list(rule.methods).sort()
        } for rule in app.url_map.iter_rules()]
    }


@app.route('/api/settings/masks')
def list_mask_filters():
    return jsonify(list(mask_filters.keys()))


@app.route('/api/settings/models')
def list_models():
    return jsonify(available_models)


@app.route('/api/settings/noises')
def list_noise_sources():
    return jsonify(list(noise_sources.keys()))


@app.route('/api/settings/params')
def list_params():
    return jsonify(config_params)


@app.route('/api/settings/platforms')
def list_platforms():
    return jsonify(list(platform_providers.keys()))


@app.route('/api/settings/schedulers')
def list_schedulers():
    return jsonify(list(pipeline_schedulers.keys()))


@app.route('/api/img2img', methods=['POST'])
def img2img():
    input_file = request.files.get('source')
    input_image = Image.open(BytesIO(input_file.read())).convert('RGB')

    strength = get_and_clamp_float(request.args, 'strength', 0.5, 1.0)

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    (output_file, output_full) = make_output_path(
        'img2img',
        seed, (
            model,
            provider,
            scheduler.__name__,
            prompt,
            negative_prompt,
            cfg,
            steps,
            strength,
            height,
            width))
    print("img2img output: %s" % (output_full))

    input_image.thumbnail((width, height))
    executor.submit_stored(output_file, run_img2img_pipeline, model, provider,
                           scheduler, prompt, negative_prompt, cfg, steps, seed, output_full, strength, input_image)

    return jsonify({
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


@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    (output_file, output_full) = make_output_path(
        'txt2img',
        seed, (
            model,
            provider,
            scheduler.__name__,
            prompt,
            negative_prompt,
            cfg,
            steps,
            height,
            width))
    print("txt2img output: %s" % (output_full))

    executor.submit_stored(output_file, run_txt2img_pipeline, model,
                           provider, scheduler, prompt, negative_prompt, cfg, steps, seed, output_full, height, width)

    return jsonify({
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


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    mask_file = request.files.get('mask')
    mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    left = get_and_clamp_int(request.args, 'left', 0,
                             config_params.get('width').get('max'), 0)
    right = get_and_clamp_int(request.args, 'right',
                              0, config_params.get('width').get('max'), 0)
    top = get_and_clamp_int(request.args, 'top', 0,
                            config_params.get('height').get('max'), 0)
    bottom = get_and_clamp_int(
        request.args, 'bottom', 0, config_params.get('height').get('max'), 0)

    mask_filter = get_from_map(request.args, 'filter', mask_filters, 'none')
    noise_source = get_from_map(
        request.args, 'noise', noise_sources, 'histogram')

    (output_file, output_full) = make_output_path(
        'inpaint', seed, (
            model,
            provider,
            scheduler.__name__,
            prompt,
            negative_prompt,
            cfg,
            steps,
            height,
            width,
            left,
            right,
            top,
            bottom,
            mask_filter.__name__,
            noise_source.__name__,
        ))
    print("inpaint output: %s" % output_full)

    source_image.thumbnail((width, height))
    mask_image.thumbnail((width, height))
    executor.submit_stored(
        output_file,
        run_inpaint_pipeline,
        model,
        provider,
        scheduler,
        prompt,
        negative_prompt,
        cfg,
        steps,
        seed,
        output_full,
        height,
        width,
        source_image,
        mask_image,
        left,
        right,
        top,
        bottom,
        noise_source,
        mask_filter)

    return jsonify({
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


@app.route('/api/ready')
def ready():
    output_file = request.args.get('output', None)
    done = executor.futures.done(output_file)

    if done == True:
        executor.futures.pop(output_file)

    return jsonify({
        'ready': done,
    })


@app.route('/api/output/<path:filename>')
def output(filename: str):
    return send_from_directory(path.join('..', output_path), filename, as_attachment=False)
