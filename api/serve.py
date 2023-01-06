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
from flask import Flask, jsonify, request, send_from_directory, url_for
from stringcase import spinalcase
from os import environ, makedirs, path, scandir
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
model_path = environ.get('ONNX_WEB_MODEL_PATH', "../models")
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', "../outputs")


# pipeline caching
available_models = []
pipeline_options = (None, None, None)
pipeline_instance = None

# pipeline params
platform_providers = {
    'amd': 'DmlExecutionProvider',
    'cpu': 'CPUExecutionProvider',
}
pipeline_schedulers = {
    'ddim': DDIMScheduler,
    'ddpm': DDPMScheduler,
    'dpm-multi': DPMSolverMultistepScheduler,
    'euler': EulerDiscreteScheduler,
    'euler-a': EulerAncestralDiscreteScheduler,
    'lms-discrete': LMSDiscreteScheduler,
    'pndm': PNDMScheduler,
}


def get_and_clamp(args, key, default_value, max_value, min_value=1):
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key, values, default):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


# from https://www.travelneil.com/stable-diffusion-updates.html
def get_latents_from_seed(seed: int, width: int, height: int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def load_pipeline(model, provider, scheduler):
    global pipeline_instance
    global pipeline_options

    options = (model, provider, scheduler)
    if pipeline_instance != None and pipeline_options == options:
        print('reusing existing pipeline')
        return pipeline_instance

    print('loading different pipeline')
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        model,
        provider=provider,
        safety_checker=None,
        scheduler=scheduler.from_pretrained(model, subfolder="scheduler")
    )
    pipeline_options = options
    pipeline_instance = pipe
    return pipe


def json_with_cors(data, origin='*'):
    """Build a JSON response with CORS headers allowing `origin`"""
    res = jsonify(data)
    res.access_control_allow_origin = origin
    return res


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


check_paths()
load_models()
app = Flask(__name__)

# routes


@app.route('/')
def index():
    return {
        'name': 'onnx-web',
        'routes': [{
            'path': url_from_rule(rule),
            'methods': list(rule.methods)
        } for rule in app.url_map.iter_rules()]
    }


@app.route('/settings/models')
def list_models():
    return json_with_cors(available_models)


@app.route('/settings/platforms')
def list_platforms():
    return json_with_cors(list(platform_providers.keys()))


@app.route('/settings/schedulers')
def list_schedulers():
    return json_with_cors(list(pipeline_schedulers.keys()))


@app.route('/txt2img')
def txt2img():
    user = request.remote_addr

    # pipeline stuff
    model = path.join(model_path, request.args.get('model'))
    provider = get_from_map(request.args, 'platform',
                            platform_providers, 'amd')
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, 'euler-a')

    # image params
    prompt = request.args.get('prompt', default_prompt)
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

    pipe = load_pipeline(model, provider, scheduler)
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

    return json_with_cors({
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


@app.route('/output/<path:filename>')
def output(filename):
    return send_from_directory(output_path, filename, as_attachment=False)
