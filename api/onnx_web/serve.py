from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_executor import Executor
from hashlib import sha256
from io import BytesIO
from PIL import Image
from struct import pack
from os import environ, makedirs, path, scandir
from typing import Tuple, Union

import json

from .pipeline import run_img2img_pipeline, run_inpaint_pipeline, run_txt2img_pipeline
from .utils import get_and_clamp_float, safer_join
from .web import pipeline_from_request, url_from_rule

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


def serve_bundle_file(filename='index.html'):
    return send_from_directory(path.join('..', bundle_path), filename)


def make_output_path(mode: str, seed: int, params: Tuple[Union[str, int, float]]):
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

    output_file = '%s_%s_%s.png' % (mode, seed, sha.hexdigest())
    output_full = safer_join(output_path, output_file)

    return (output_file, output_full)


def get_model_path(model: str):
    return safer_join(model_path, model)


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


@app.route('/api/settings/models')
def list_models():
    return jsonify(available_models)


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
    input_image.thumbnail((default_width, default_height))

    strength = get_and_clamp_float(request.args, 'strength', 0.5, 1.0)

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    (output_file, output_full) = make_output_path('img2img', seed,
                                                  (prompt, cfg, negative_prompt, steps, strength, height, width))
    print("img2img output: %s" % (output_full))

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
            'height': default_height,
            'width': default_width,
        }
    })


@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    (output_file, output_full) = make_output_path('txt2img',
                                                  seed, (prompt, cfg, negative_prompt, steps, height, width))
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
    source_image.thumbnail((default_width, default_height))

    mask_file = request.files.get('mask')
    mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')
    mask_image.thumbnail((default_width, default_height))

    (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height,
     width, seed) = pipeline_from_request()

    (output_file, output_full) = make_output_path(
        'inpaint', seed, (prompt, cfg, steps, height, width, seed))
    print("inpaint output: %s" % output_full)

    executor.submit_stored(output_file, run_inpaint_pipeline, model, provider, scheduler, prompt, negative_prompt,
                           cfg, steps, seed, output_full, height, width, source_image, mask_image)

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
            'height': default_height,
            'width': default_width,
        }
    })


@app.route('/api/ready')
def ready():
    output_file = request.args.get('output', None)

    return jsonify({
        'ready': executor.futures.done(output_file),
    })


@app.route('/api/output/<path:filename>')
def output(filename: str):
    return send_from_directory(path.join('..', output_path), filename, as_attachment=False)
