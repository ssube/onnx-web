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
from flask import request, url_for
import numpy as np

from .serve import get_model_path
from .utils import get_and_clamp_float, get_and_clamp_int, get_from_map

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

def url_from_rule(rule):
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)

def pipeline_from_request(config_params):
    user = request.remote_addr

    # pipeline stuff
    model = get_model_path(request.args.get('model', config_params.get('model').get('default')))
    provider = get_from_map(request.args, 'platform',
                            platform_providers, config_params.get('platform').get('default'))
    scheduler = get_from_map(request.args, 'scheduler',
                             pipeline_schedulers, config_params.get('scheduler').get('default'))

    # image params
    prompt = request.args.get('prompt', config_params.get('prompt').get('default'))
    negative_prompt = request.args.get('negativePrompt', None)

    if negative_prompt is not None and negative_prompt.strip() == '':
        negative_prompt = None

    cfg = get_and_clamp_float(
        request.args, 'cfg', config_params.get('cfg').get('default'), config_params.get('cfg').get('max'), 0)
    steps = get_and_clamp_int(
        request.args, 'steps', config_params.get('steps').get('default'), config_params.get('steps').get('max'))
    height = get_and_clamp_int(
        request.args, 'height', config_params.get('height').get('default'), config_params.get('height').get('max'))
    width = get_and_clamp_int(
        request.args, 'width', config_params.get('width').get('default'), config_params.get('width').get('max'))

    seed = int(request.args.get('seed', -1))
    if seed == -1:
        seed = np.random.randint(np.iinfo(np.int32).max)

    print("request from %s: %s rounds of %s using %s on %s, %sx%s, %s, %s - %s" %
          (user, steps, scheduler.__name__, model, provider, width, height, cfg, seed, prompt))

    return (model, provider, scheduler, prompt, negative_prompt, cfg, steps, height, width, seed)

