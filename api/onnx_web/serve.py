from . import logging
from diffusers import (
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
from flask import Flask, jsonify, make_response, request, send_from_directory, url_for
from flask_cors import CORS
from glob import glob
from io import BytesIO
from jsonschema import validate
from logging import getLogger
from PIL import Image
from onnxruntime import get_available_providers
from os import makedirs, path
from typing import List, Tuple


from .chain import (
    blend_img2img,
    blend_inpaint,
    correct_gfpgan,
    persist_disk,
    persist_s3,
    reduce_thumbnail,
    reduce_crop,
    source_noise,
    source_txt2img,
    upscale_outpaint,
    upscale_resrgan,
    upscale_stable_diffusion,
    ChainPipeline,
)
from .device_pool import (
    DeviceParams,
    DevicePoolExecutor,
)
from .diffusion.run import (
    run_img2img_pipeline,
    run_inpaint_pipeline,
    run_txt2img_pipeline,
    run_upscale_pipeline,
)
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
from .output import (
    json_params,
    make_output_name,
)
from .params import (
    Border,
    DeviceParams,
    ImageParams,
    Size,
    StageParams,
    UpscaleParams,
)
from .utils import (
    base_join,
    is_debug,
    get_and_clamp_float,
    get_and_clamp_int,
    get_from_list,
    get_from_map,
    get_not_empty,
    get_size,
    ServerContext,
)

import gc
import numpy as np
import torch
import yaml

logger = getLogger(__name__)

# config caching
config_params = {}

# pipeline params
platform_providers = {
    'amd': 'DmlExecutionProvider',
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'directml': 'DmlExecutionProvider',
    'nvidia': 'CUDAExecutionProvider',
    'rocm': 'ROCMExecutionProvider',
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
chain_stages = {
    'blend-img2img': blend_img2img,
    'blend-inpaint': blend_inpaint,
    'correct-gfpgan': correct_gfpgan,
    'persist-disk': persist_disk,
    'persist-s3': persist_s3,
    'reduce-crop': reduce_crop,
    'reduce-thumbnail': reduce_thumbnail,
    'source-noise': source_noise,
    'source-txt2img': source_txt2img,
    'upscale-outpaint': upscale_outpaint,
    'upscale-resrgan': upscale_resrgan,
    'upscale-stable-diffusion': upscale_stable_diffusion,
}

# Available ORT providers
available_platforms: List[DeviceParams] = []

# loaded from model_path
diffusion_models = []
correction_models = []
upscaling_models = []


def get_config_value(key: str, subkey: str = 'default'):
    return config_params.get(key).get(subkey)


def url_from_rule(rule) -> str:
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)


def pipeline_from_request() -> Tuple[DeviceParams, ImageParams, Size]:
    user = request.remote_addr

    # platform stuff
    device_name = request.args.get('platform', available_platforms[0].device)
    device = None

    for platform in available_platforms:
        if platform.device == device_name:
            device = available_platforms[0]

    if device is None:
        raise Exception('unknown device')

    # pipeline stuff
    model = get_not_empty(request.args, 'model', get_config_value('model'))
    model_path = get_model_path(model)
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, get_config_value('scheduler'))

    # image params
    prompt = get_not_empty(request.args,
                           'prompt', get_config_value('prompt'))
    negative_prompt = request.args.get('negativePrompt', None)

    if negative_prompt is not None and negative_prompt.strip() == '':
        negative_prompt = None

    cfg = get_and_clamp_float(
        request.args, 'cfg',
        get_config_value('cfg'),
        get_config_value('cfg', 'max'),
        get_config_value('cfg', 'min'))
    steps = get_and_clamp_int(
        request.args, 'steps',
        get_config_value('steps'),
        get_config_value('steps', 'max'),
        get_config_value('steps', 'min'))
    height = get_and_clamp_int(
        request.args, 'height',
        get_config_value('height'),
        get_config_value('height', 'max'),
        get_config_value('height', 'min'))
    width = get_and_clamp_int(
        request.args, 'width',
        get_config_value('width'),
        get_config_value('width', 'max'),
        get_config_value('width', 'min'))

    seed = int(request.args.get('seed', -1))
    if seed == -1:
        seed = np.random.randint(np.iinfo(np.int32).max)

    logger.info("request from %s: %s rounds of %s using %s on %s, %sx%s, %s, %s - %s",
                user, steps, scheduler.__name__, model_path, device.provider, width, height, cfg, seed, prompt)

    params = ImageParams(model_path, device.provider, scheduler, prompt,
                         negative_prompt, cfg, steps, seed)
    size = Size(width, height)
    return (device, params, size)


def border_from_request() -> Border:
    left = get_and_clamp_int(request.args, 'left', 0,
                             get_config_value('width', 'max'), 0)
    right = get_and_clamp_int(request.args, 'right',
                              0, get_config_value('width', 'max'), 0)
    top = get_and_clamp_int(request.args, 'top', 0,
                            get_config_value('height', 'max'), 0)
    bottom = get_and_clamp_int(
        request.args, 'bottom', 0, get_config_value('height', 'max'), 0)

    return Border(left, right, top, bottom)


def upscale_from_request(provider: str) -> UpscaleParams:
    denoise = get_and_clamp_float(request.args, 'denoise', 0.5, 1.0, 0.0)
    scale = get_and_clamp_int(request.args, 'scale', 1, 4, 1)
    outscale = get_and_clamp_int(request.args, 'outscale', 1, 4, 1)
    upscaling = get_from_list(request.args, 'upscaling', upscaling_models)
    correction = get_from_list(request.args, 'correction', correction_models)
    faces = get_not_empty(request.args, 'faces', 'false') == 'true'
    face_strength = get_and_clamp_float(
        request.args, 'faceStrength', 0.5, 1.0, 0.0)

    return UpscaleParams(
        upscaling,
        provider,
        correction_model=correction,
        denoise=denoise,
        faces=faces,
        face_strength=face_strength,
        format='onnx',
        outscale=outscale,
        scale=scale,
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
    diffusion_models.sort()

    correction_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, 'correction-*'))]
    correction_models = list(set(correction_models))
    correction_models.sort()

    upscaling_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, 'upscaling-*'))]
    upscaling_models = list(set(upscaling_models))
    upscaling_models.sort()


def load_params(context: ServerContext):
    global config_params
    params_file = path.join(context.params_path, 'params.json')
    with open(params_file, 'r') as f:
        config_params = yaml.safe_load(f)

        if 'platform' in config_params and context.default_platform is not None:
            logger.info('overriding default platform to %s',
                        context.default_platform)
            config_platform = config_params.get('platform')
            config_platform['default'] = context.default_platform


def load_platforms():
    global available_platforms

    providers = get_available_providers()

    for potential in platform_providers:
        if platform_providers[potential] in providers and potential not in context.block_platforms:
            if potential == 'cuda':
                for i in range(torch.cuda.device_count()):
                    available_platforms.append(DeviceParams('%s:%s' % (potential, i), platform_providers[potential], {
                        'device_id': i,
                    }))
            else:
                available_platforms.append(DeviceParams(
                    potential, platform_providers[potential]))

    logger.info('available acceleration platforms: %s',
                ', '.join([str(p) for p in available_platforms]))


context = ServerContext.from_environ()
check_paths(context)
load_models(context)
load_params(context)
load_platforms()

app = Flask(__name__)
CORS(app, origins=context.cors_origin)

executor = DevicePoolExecutor(available_platforms)

if is_debug():
    gc.set_debug(gc.DEBUG_STATS)


def ready_reply(ready: bool, progress: int = 0):
    return jsonify({
        'progress': progress,
        'ready': ready,
    })


def error_reply(err: str):
    response = make_response(jsonify({
        'error': err,
    }))
    response.status_code = 400
    return response


def get_model_path(model: str):
    return base_join(context.model_path, model)


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
    return jsonify([p.device for p in available_platforms])


@app.route('/api/settings/schedulers')
def list_schedulers():
    return jsonify(list(pipeline_schedulers.keys()))


@app.route('/api/img2img', methods=['POST'])
def img2img():
    if 'source' not in request.files:
        return error_reply('source image is required')

    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    params, size = pipeline_from_request()
    upscale = upscale_from_request(params.provider)

    strength = get_and_clamp_float(
        request.args,
        'strength',
        get_config_value('strength'),
        get_config_value('strength', 'max'),
        get_config_value('strength', 'min'))

    output = make_output_name(
        context,
        'img2img',
        params,
        size,
        extras=(strength,))
    logger.info("img2img job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    executor.submit(output, run_img2img_pipeline,
                    context, params, output, upscale, source_image, strength)

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    params, size = pipeline_from_request()
    upscale = upscale_from_request(params.provider)

    output = make_output_name(
        context,
        'txt2img',
        params,
        size)
    logger.info("txt2img job queued for: %s", output)

    executor.submit(
        output, run_txt2img_pipeline, context, params, size, output, upscale)

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    if 'source' not in request.files:
        return error_reply('source image is required')

    if 'mask' not in request.files:
        return error_reply('mask image is required')

    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    mask_file = request.files.get('mask')
    mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')

    params, size = pipeline_from_request()
    expand = border_from_request()
    upscale = upscale_from_request(params.provider)

    fill_color = get_not_empty(request.args, 'fillColor', 'white')
    mask_filter = get_from_map(request.args, 'filter', mask_filters, 'none')
    noise_source = get_from_map(
        request.args, 'noise', noise_sources, 'histogram')
    strength = get_and_clamp_float(
        request.args,
        'strength',
        get_config_value('strength'),
        get_config_value('strength', 'max'),
        get_config_value('strength', 'min'))

    output = make_output_name(
        context,
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
            strength,
            fill_color,
        )
    )
    logger.info("inpaint job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    mask_image.thumbnail((size.width, size.height))
    executor.submit(
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
        mask_filter,
        strength,
        fill_color)

    return jsonify(json_params(output, params, size, upscale=upscale, border=expand))


@app.route('/api/upscale', methods=['POST'])
def upscale():
    if 'source' not in request.files:
        return error_reply('source image is required')

    source_file = request.files.get('source')
    source_image = Image.open(BytesIO(source_file.read())).convert('RGB')

    params, size = pipeline_from_request()
    upscale = upscale_from_request(params.provider)

    output = make_output_name(
        context,
        'upscale',
        params,
        size)
    logger.info("upscale job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    executor.submit(output, run_upscale_pipeline,
                    context, params, size, output, upscale, source_image)

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route('/api/chain', methods=['POST'])
def chain():
    logger.debug('chain pipeline request: %s, %s',
                 request.form.keys(), request.files.keys())
    body = request.form.get('chain') or request.files.get('chain')
    if body is None:
        return error_reply('chain pipeline must have a body')

    data = yaml.safe_load(body)
    with open('./schema.yaml', 'r') as f:
        schema = yaml.safe_load(f.read())

    logger.info('validating chain request: %s against %s', data, schema)
    validate(data, schema)

    # get defaults from the regular parameters
    params, size = pipeline_from_request()
    output = make_output_name(
        context,
        'chain',
        params,
        size)

    pipeline = ChainPipeline()
    for stage_data in data.get('stages', []):
        callback = chain_stages[stage_data.get('type')]
        kwargs = stage_data.get('params', {})
        logger.info('request stage: %s, %s', callback.__name__, kwargs)

        stage = StageParams(
            stage_data.get('name', callback.__name__),
            tile_size=get_size(kwargs.get('tile_size')),
            outscale=get_and_clamp_int(kwargs, 'outscale', 1, 4),
        )

        if 'border' in kwargs:
            border = Border.even(int(kwargs.get('border')))
            kwargs['border'] = border

        if 'upscale' in kwargs:
            upscale = UpscaleParams(kwargs.get('upscale'), params.provider)
            kwargs['upscale'] = upscale

        stage_source_name = 'source:%s' % (stage.name)
        stage_mask_name = 'mask:%s' % (stage.name)

        if stage_source_name in request.files:
            logger.debug('loading source image %s for pipeline stage %s',
                         stage_source_name, stage.name)
            source_file = request.files.get(stage_source_name)
            source_image = Image.open(
                BytesIO(source_file.read())).convert('RGB')
            source_image = source_image.thumbnail((512, 512))
            kwargs['source_image'] = source_image

        if stage_mask_name in request.files:
            logger.debug('loading mask image %s for pipeline stage %s',
                         stage_mask_name, stage.name)
            mask_file = request.files.get(stage_mask_name)
            mask_image = Image.open(BytesIO(mask_file.read())).convert('RGB')
            mask_image = mask_image.thumbnail((512, 512))
            kwargs['mask_image'] = mask_image

        pipeline.append((callback, stage, kwargs))

    logger.info('running chain pipeline with %s stages', len(pipeline.stages))

    # build and run chain pipeline
    empty_source = Image.new('RGB', (size.width, size.height))
    executor.submit(output, pipeline, context,
                    params, empty_source, output=output, size=size)

    return jsonify(json_params(output, params, size))


@app.route('/api/cancel', methods=['PUT'])
def cancel():
    output_file = request.args.get('output', None)

    cancel = executor.cancel(output_file)

    return ready_reply(cancel)


@app.route('/api/ready')
def ready():
    output_file = request.args.get('output', None)

    done, progress = executor.done(output_file)

    if done is None:
        file = base_join(context.output_path, output_file)
        if path.exists(file):
            return ready_reply(True)

    return ready_reply(done, progress=progress)


@app.route('/api/status')
def status():
    return jsonify(executor.status())


@app.route('/output/<path:filename>')
def output(filename: str):
    return send_from_directory(path.join('..', context.output_path), filename, as_attachment=False)
