# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from logging import getLogger
from math import ceil
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import PIL_INTERPOLATION, deprecate
from transformers import CLIPImageProcessor, CLIPTokenizer

from ...chain.tile import make_tile_mask
from ...constants import LATENT_CHANNELS, LATENT_FACTOR
from ...params import Size
from ..utils import (
    expand_latents,
    parse_regions,
    random_seed,
    repair_nan,
    resize_latent_shape,
)
from .base import OnnxStableDiffusionBasePipeline

logger = getLogger(__name__)


# inpaint constants
NUM_UNET_INPUT_CHANNELS = 9

DEFAULT_WINDOW = 32
DEFAULT_STRIDE = 8


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def prepare_mask_and_masked_image(image, mask, latents_shape):
    image = np.array(
        image.convert("RGB").resize((latents_shape[1] * 8, latents_shape[0] * 8))
    )
    image = image[None].transpose(0, 3, 1, 2)
    image = image.astype(np.float32) / 127.5 - 1.0

    image_mask = np.array(
        mask.convert("L").resize((latents_shape[1] * 8, latents_shape[0] * 8))
    )
    masked_image = image * (image_mask < 127.5)

    mask = mask.resize(
        (latents_shape[1], latents_shape[0]), PIL_INTERPOLATION["nearest"]
    )
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    return mask, masked_image


class OnnxStableDiffusionPanoramaPipeline(OnnxStableDiffusionBasePipeline):
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: OnnxRuntimeModel,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        window: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        super().__init__(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        self.window = window or DEFAULT_WINDOW
        self.stride = stride or DEFAULT_STRIDE

    def get_views(
        self, panorama_height: int, panorama_width: int, window_size: int, stride: int
    ) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        panorama_height /= 8
        panorama_width /= 8

        num_blocks_height = ceil(abs((panorama_height - window_size) / stride)) + 1
        num_blocks_width = ceil(abs((panorama_width - window_size) / stride)) + 1

        total_num_blocks = int(num_blocks_height * num_blocks_width)
        logger.debug(
            "panorama generated %s views, %s by %s blocks",
            total_num_blocks,
            num_blocks_height,
            num_blocks_width,
        )

        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))

        return (views, (h_end * 8, w_end * 8))

    @torch.no_grad()
    def text2img(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image` or List[`PIL.Image.Image`] or `torch.FloatTensor`):
                `Image`, or tensor representing an image batch which will be upscaled. *
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                One or a list of [numpy generator(s)](TODO) to make generation deterministic.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt, regions = parse_regions(prompt)

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3.b. Encode region prompts
        region_embeds: List[np.ndarray] = []

        for _top, _left, _bottom, _right, _weight, _feather, region_prompt in regions:
            if region_prompt.endswith("+"):
                region_prompt = region_prompt[:-1] + " " + prompt

            region_prompt_embeds = self._encode_prompt(
                region_prompt,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )

            region_embeds.append(region_prompt_embeds)

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (
            batch_size * num_images_per_prompt,
            LATENT_CHANNELS,
            height // LATENT_FACTOR,
            width // LATENT_FACTOR,
        )
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (
                input.type
                for input in self.unet.model.get_inputs()
                if input.name == "timestep"
            ),
            "tensor(float)",
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # panorama additions
        views, resize = self.get_views(height, width, self.window, self.stride)
        logger.trace("panorama resized latents to %s", resize)

        count = np.zeros(resize_latent_shape(latents, resize))
        value = np.zeros(resize_latent_shape(latents, resize))

        # adjust latents
        latents = expand_latents(
            latents,
            random_seed(generator),
            Size(resize[1], resize[0]),
            sigma=self.scheduler.init_noise_sigma,
        )

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            last = i == (len(self.scheduler.timesteps) - 1)
            count.fill(0)
            value.fill(0)

            for h_start, h_end, w_start, w_end in views:
                # get the latents corresponding to the current view coordinates
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    np.concatenate([latents_for_view] * 2)
                    if do_classifier_free_guidance
                    else latents_for_view
                )
                latent_model_input = self.scheduler.scale_model_input(
                    torch.from_numpy(latent_model_input), t
                )
                latent_model_input = latent_model_input.cpu().numpy()

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                )
                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred),
                    t,
                    torch.from_numpy(latents_for_view),
                    **extra_step_kwargs,
                )
                latents_view_denoised = scheduler_output.prev_sample.numpy()

                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1

            if not last:
                for r, region in enumerate(regions):
                    top, left, bottom, right, weight, feather, prompt = region
                    logger.debug(
                        "running region prompt: %s, %s, %s, %s, %s, %s, %s",
                        top,
                        left,
                        bottom,
                        right,
                        weight,
                        feather,
                        prompt,
                    )

                    # convert coordinates to latent space
                    h_start = top // LATENT_FACTOR
                    h_end = bottom // LATENT_FACTOR
                    w_start = left // LATENT_FACTOR
                    w_end = right // LATENT_FACTOR

                    # get the latents corresponding to the current view coordinates
                    latents_for_region = latents[:, :, h_start:h_end, w_start:w_end]
                    logger.trace(
                        "region latent shape: [:,:,%s:%s,%s:%s] -> %s",
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                        latents_for_region.shape,
                    )

                    # expand the latents if we are doing classifier free guidance
                    latent_region_input = (
                        np.concatenate([latents_for_region] * 2)
                        if do_classifier_free_guidance
                        else latents_for_region
                    )
                    latent_region_input = self.scheduler.scale_model_input(
                        torch.from_numpy(latent_region_input), t
                    )
                    latent_region_input = latent_region_input.cpu().numpy()

                    # predict the noise residual
                    timestep = np.array([t], dtype=timestep_dtype)
                    region_noise_pred = self.unet(
                        sample=latent_region_input,
                        timestep=timestep,
                        encoder_hidden_states=region_embeds[r],
                    )
                    region_noise_pred = region_noise_pred[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        region_noise_pred_uncond, region_noise_pred_text = np.split(
                            region_noise_pred, 2
                        )
                        region_noise_pred = (
                            region_noise_pred_uncond
                            + guidance_scale
                            * (region_noise_pred_text - region_noise_pred_uncond)
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_output = self.scheduler.step(
                        torch.from_numpy(region_noise_pred),
                        t,
                        torch.from_numpy(latents_for_region),
                        **extra_step_kwargs,
                    )
                    latents_region_denoised = scheduler_output.prev_sample.numpy()

                    if feather[0] > 0.0:
                        mask = make_tile_mask(
                            (h_end - h_start, w_end - w_start),
                            (h_end - h_start, w_end - w_start),
                            feather[0],
                            feather[1],
                        )
                        mask = np.expand_dims(mask, axis=0)
                        mask = np.repeat(mask, 4, axis=0)
                        mask = np.expand_dims(mask, axis=0)
                    else:
                        mask = 1

                    if weight >= 100.0:
                        value[:, :, h_start:h_end, w_start:w_end] = (
                            latents_region_denoised * mask
                        )
                        count[:, :, h_start:h_end, w_start:w_end] = mask
                    else:
                        value[:, :, h_start:h_end, w_start:w_end] += (
                            latents_region_denoised * weight * mask
                        )
                        count[:, :, h_start:h_end, w_start:w_end] += weight * mask

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = np.where(count > 0, value / count, value)
            latents = repair_nan(latents)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # remove extra margins
        latents = latents[
            :, :, 0 : (height // LATENT_FACTOR), 0 : (width // LATENT_FACTOR)
        ]

        latents = np.clip(latents, -4, +4)
        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image` or List[`PIL.Image.Image`] or `torch.FloatTensor`):
                `Image`, or tensor representing an image batch which will be upscaled. *
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                One or a list of [numpy generator(s)](TODO) to make generation deterministic.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        height = image.height
        width = image.width

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if generator is None:
            generator = np.random

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # prep image
        image = preprocess(image).cpu().numpy()

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        image = image.astype(latents_dtype)

        # encode the init image into latents and scale the latents
        latents = self.vae_encoder(sample=image)[0]
        latents = 0.18215 * latents

        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) > latents.shape[0] and len(prompt) % latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {len(prompt)} text prompts (`prompt`), but only {latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = len(prompt) // latents.shape[0]
            latents = np.concatenate(
                [latents] * additional_image_per_prompt * num_images_per_prompt, axis=0
            )
        elif len(prompt) > latents.shape[0] and len(prompt) % latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {latents.shape[0]} to {len(prompt)} text prompts."
            )
        else:
            latents = np.concatenate([latents] * num_images_per_prompt, axis=0)

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
        timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)

        noise = generator.randn(*latents.shape).astype(latents_dtype)
        latents = self.scheduler.add_noise(
            torch.from_numpy(latents),
            torch.from_numpy(noise),
            torch.from_numpy(timesteps),
        )
        latents = latents.numpy()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].numpy()

        timestep_dtype = next(
            (
                input.type
                for input in self.unet.model.get_inputs()
                if input.name == "timestep"
            ),
            "tensor(float)",
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # panorama additions
        views, resize = self.get_views(height, width, self.window, self.stride)
        logger.trace("panorama resized latents to %s", resize)

        count = np.zeros(resize_latent_shape(latents, resize))
        value = np.zeros(resize_latent_shape(latents, resize))

        # adjust latents
        latents = expand_latents(
            latents,
            random_seed(generator),
            Size(resize[1], resize[0]),
            sigma=self.scheduler.init_noise_sigma,
        )

        for i, t in enumerate(self.progress_bar(timesteps)):
            count.fill(0)
            value.fill(0)

            for h_start, h_end, w_start, w_end in views:
                # get the latents corresponding to the current view coordinates
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    np.concatenate([latents_for_view] * 2)
                    if do_classifier_free_guidance
                    else latents_for_view
                )
                latent_model_input = self.scheduler.scale_model_input(
                    torch.from_numpy(latent_model_input), t
                )
                latent_model_input = latent_model_input.cpu().numpy()

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                )
                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred),
                    t,
                    torch.from_numpy(latents_for_view),
                    **extra_step_kwargs,
                )
                latents_view_denoised = scheduler_output.prev_sample.numpy()

                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = np.where(count > 0, value / count, value)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # remove extra margins
        latents = latents[
            :, :, 0 : (height // LATENT_FACTOR), 0 : (width // LATENT_FACTOR)
        ]

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def inpaint(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                A np.random.RandomState to make generation deterministic.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: np.ndarray)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        num_channels_latents = LATENT_CHANNELS
        latents_shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // LATENT_FACTOR,
            width // LATENT_FACTOR,
        )
        latents_dtype = prompt_embeds.dtype
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        # prepare mask and masked_image
        mask, masked_image = prepare_mask_and_masked_image(
            image, mask_image, latents_shape[-2:]
        )
        mask = mask.astype(latents.dtype)
        masked_image = masked_image.astype(latents.dtype)

        masked_image_latents = self.vae_encoder(sample=masked_image)[0]
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt
        mask = mask.repeat(batch_size * num_images_per_prompt, 0)
        masked_image_latents = masked_image_latents.repeat(
            batch_size * num_images_per_prompt, 0
        )

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            np.concatenate([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]

        unet_input_channels = NUM_UNET_INPUT_CHANNELS
        if (
            num_channels_latents + num_channels_mask + num_channels_masked_image
            != unet_input_channels
        ):
            raise ValueError(
                "Incorrect configuration settings! The config of `pipeline.unet` expects"
                f" {unet_input_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (
                input.type
                for input in self.unet.model.get_inputs()
                if input.name == "timestep"
            ),
            "tensor(float)",
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # panorama additions
        views, resize = self.get_views(height, width, self.window, self.stride)
        logger.trace("panorama resized latents to %s", resize)

        count = np.zeros(resize_latent_shape(latents, resize))
        value = np.zeros(resize_latent_shape(latents, resize))

        # adjust latents
        latents = expand_latents(
            latents,
            random_seed(generator),
            Size(resize[1], resize[0]),
            sigma=self.scheduler.init_noise_sigma,
        )

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            count.fill(0)
            value.fill(0)

            for h_start, h_end, w_start, w_end in views:
                # get the latents corresponding to the current view coordinates
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]
                mask_for_view = mask[:, :, h_start:h_end, w_start:w_end]
                masked_latents_for_view = masked_image_latents[
                    :, :, h_start:h_end, w_start:w_end
                ]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    np.concatenate([latents_for_view] * 2)
                    if do_classifier_free_guidance
                    else latents_for_view
                )
                # concat latents, mask, masked_image_latnets in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(
                    torch.from_numpy(latent_model_input), t
                )
                latent_model_input = latent_model_input.cpu().numpy()
                latent_model_input = np.concatenate(
                    [latent_model_input, mask_for_view, masked_latents_for_view], axis=1
                )

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred),
                    t,
                    torch.from_numpy(latents_for_view),
                    **extra_step_kwargs,
                )
                latents_view_denoised = scheduler_output.prev_sample.numpy()

                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = np.where(count > 0, value / count, value)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # remove extra margins
        latents = latents[
            :, :, 0 : (height // LATENT_FACTOR), 0 : (width // LATENT_FACTOR)
        ]

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)
            # safety_checker does not support batched inputs yet
            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        if "image" in kwargs or (
            len(args) > 1
            and (
                isinstance(args[1], np.ndarray) or isinstance(args[1], PIL.Image.Image)
            )
        ):
            if "mask_image" in kwargs or (
                len(args) > 2
                and (
                    isinstance(args[1], np.ndarray)
                    or isinstance(args[1], PIL.Image.Image)
                )
            ):
                logger.debug("running inpaint panorama pipeline")
                return self.inpaint(*args, **kwargs)
            else:
                logger.debug("running img2img panorama pipeline")
                return self.img2img(*args, **kwargs)
        else:
            logger.debug("running txt2img panorama pipeline")
            return self.text2img(*args, **kwargs)

    def set_window_size(self, window: int, stride: int):
        self.window = window
        self.stride = stride
