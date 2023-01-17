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
)
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from flask_executor import Executor
from glob import glob
from io import BytesIO
from PIL import Image
from os import makedirs, path, scandir
from typing import Tuple

from .image import (
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
from .pipeline import (
    run_img2img_pipeline,
    run_inpaint_pipeline,
    run_txt2img_pipeline,
)
from .upscale import (
    UpscaleParams,
)
from .utils import (
    get_and_clamp_float,
    get_and_clamp_int,
    get_from_list,
    get_from_map,
    make_output_name,
    safer_join,
    BaseParams,
    Border,
    ServerContext,
    Size,
)

import json
import numpy as np

# pipeline caching
config_params = {}

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

# loaded from model_path
diffusion_models = []
correction_models = []
upscaling_models = []


def url_from_rule(rule) -> str:
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)


def pipeline_from_request() -> Tuple[BaseParams, Size]:
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

    params = BaseParams(model, provider, scheduler, prompt,
                        negative_prompt, cfg, steps, seed)
    size = Size(width, height)
    return (params, size)


def border_from_request() -> Border:
    left = get_and_clamp_int(request.args, 'left', 0,
                             config_params.get('width').get('max'), 0)
    right = get_and_clamp_int(request.args, 'right',
                              0, config_params.get('width').get('max'), 0)
    top = get_and_clamp_int(request.args, 'top', 0,
                            config_params.get('height').get('max'), 0)
    bottom = get_and_clamp_int(
        request.args, 'bottom', 0, config_params.get('height').get('max'), 0)

    return Border(left, right, top, bottom)


def upscale_from_request() -> UpscaleParams:
    denoise = get_and_clamp_float(request.args, 'denoise', 0.5, 1.0, 0.0)
    scale = get_and_clamp_int(request.args, 'scale', 1, 4, 1)
    outscale = get_and_clamp_int(request.args, 'outscale', 1, 4, 1)
    upscaling = get_from_list(request.args, 'upscaling', upscaling_models)
    correction = get_from_list(request.args, 'correction', correction_models)
    faces = request.args.get('faces', 'false') == 'true'

    return UpscaleParams(
        upscaling,
        correction_model=correction,
        scale=scale,
        outscale=outscale,
        faces=faces,
        platform='onnx',
        denoise=denoise,
    )


def check_paths(context: ServerContext):
    if not path.exists(context.model_path):
        raise RuntimeError('model path must exist')

    if not path.exists(context.output_path):
        makedirs(context.output_path)


def get_model_name(model: str) -> str:
    base = path.basename(model)
    (file, _ext) = path.splitext(base)
    return file


def load_models(context: ServerContext):
    global diffusion_models
    global correction_models
    global upscaling_models

    diffusion_models = [get_model_name(f) for f in glob(
        path.join(context.model_path, 'diffusion-*'))]
    diffusion_models.extend([
        get_model_name(f) for f in glob(path.join(context.model_path, 'stable-diffusion-*'))])
    diffusion_models = list(set(diffusion_models))

    correction_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, 'correction-*'))]
    correction_models = list(set(correction_models))

    upscaling_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, 'upscaling-*'))]
    upscaling_models = list(set(upscaling_models))


def load_params(context: ServerContext):
    global config_params
    params_file = path.join(context.params_path, 'params.json')
    with open(params_file) as f:
        config_params = json.load(f)


context = ServerContext.from_environ()

check_paths(context)
load_models(context)
load_params(context)

app = Flask(__name__)
app.config['EXECUTOR_MAX_WORKERS'] = context.num_workers
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True

CORS(app, origins=context.cors_origin)
executor = Executor(app)


# TODO: these two use context

def get_model_path(model: str):
    return safer_join(context.model_path, model)


def serve_bundle_file(filename='index.html'):
    return send_from_directory(path.join('..', context.bundle_path), filename)


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
    return jsonify({
        'diffusion': diffusion_models,
        'correction': correction_models,
        'upscaling': upscaling_models,
    })


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
    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    params, size = pipeline_from_request()
    upscale = upscale_from_request()

    strength = get_and_clamp_float(
        request.args,
        'strength',
        config_params.get('strength').get('default'),
        config_params.get('strength').get('max'))

    output = make_output_name(
        'img2img',
        params,
        size,
        extras=(strength))
    print("img2img output: %s" % (output))

    source_image.thumbnail((size.width, size.height))
    executor.submit_stored(output, run_img2img_pipeline,
                           context, params, output, upscale, source_image, strength)

    return jsonify({
        'output': output,
        'params': params.tojson(),
        'size': upscale.resize(size).tojson(),
    })


@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    params, size = pipeline_from_request()
    upscale = upscale_from_request()

    output = make_output_name(
        'txt2img',
        params,
        size)
    print("txt2img output: %s" % (output))

    executor.submit_stored(
        output, run_txt2img_pipeline, context, params, size, output, upscale)

    return jsonify({
        'output': output,
        'params': params.tojson(),
        'size': upscale.resize(size).tojson(),
    })


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    mask_file = request.files.get('mask')
    mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')

    params, size = pipeline_from_request()
    expand = border_from_request()
    upscale = upscale_from_request()

    mask_filter = get_from_map(request.args, 'filter', mask_filters, 'none')
    noise_source = get_from_map(
        request.args, 'noise', noise_sources, 'histogram')

    output = make_output_name(
        'inpaint',
        params,
        size,
        extras=(
            expand.left,
            expand.right,
            expand.top,
            expand.bottom,
            mask_filter.__name__,
            noise_source.__name__,
        )
    )
    print("inpaint output: %s" % output)

    source_image.thumbnail((size.width, size.height))
    mask_image.thumbnail((size.width, size.height))
    executor.submit_stored(
        output,
        run_inpaint_pipeline,
        context,
        params,
        size,
        output,
        upscale,
        source_image,
        mask_image,
        expand,
        noise_source,
        mask_filter)

    return jsonify({
        'output': output,
        'params': params.tojson(),
        'size': upscale.resize(size.add_border(expand)).tojson(),
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
    return send_from_directory(path.join('..', context.output_path), filename, as_attachment=False)
