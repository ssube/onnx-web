import inspect
from logging import getLogger
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from transformers import CLIPImageProcessor, CLIPTokenizer

from ...constants import LATENT_CHANNELS, LATENT_FACTOR, ONNX_MODEL
from ...convert.utils import onnx_export
from .base import OnnxStableDiffusionBasePipeline

logger = getLogger(__name__)


class OnnxStableDiffusionHighresPipeline(OnnxStableDiffusionBasePipeline):
    upscaler: OnnxRuntimeModel

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
        upscaler: OnnxRuntimeModel = None,
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

        self.upscaler = upscaler

    @torch.no_grad()
    def text2img(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        num_upscale_steps: Optional[int] = 50,
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

        prompt_embeds, text_pooler_out = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

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

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
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
                torch.from_numpy(latents),
                **extra_step_kwargs,
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        if self.upscaler is not None:
            # 5. set upscale timesteps
            self.scheduler.set_timesteps(num_upscale_steps)
            timesteps = self.scheduler.timesteps

            batch_multiplier = 2 if do_classifier_free_guidance else 1
            image = np.concatenate([latents] * batch_multiplier)

            # 5. Add noise to image (set to be 0):
            # (see below notes from the author):
            # "the This step theoretically can make the model work better on out-of-distribution inputs, but mostly
            # just seems to make it match the input less, so it's turned off by default."
            noise_level = np.array([0.0], dtype=np.float32)
            noise_level = np.concatenate([noise_level] * image.shape[0])
            inv_noise_level = (noise_level**2 + 1) ** (-0.5)

            image_cond = (
                F.interpolate(torch.tensor(image), scale_factor=2, mode="nearest")
                * inv_noise_level[:, None, None, None]
            )
            image_cond = image_cond.numpy().astype(prompt_embeds.dtype)

            noise_level_embed = np.concatenate(
                [
                    np.ones(
                        (text_pooler_out.shape[0], 64), dtype=text_pooler_out.dtype
                    ),
                    np.zeros(
                        (text_pooler_out.shape[0], 64), dtype=text_pooler_out.dtype
                    ),
                ],
                axis=1,
            )

            # upscaling latents
            latents_shape = (
                batch_size * num_images_per_prompt,
                LATENT_CHANNELS,
                height * 2 // LATENT_FACTOR,
                width * 2 // LATENT_FACTOR,
            )
            latents = generator.randn(*latents_shape).astype(latents_dtype)

            timestep_condition = np.concatenate(
                [noise_level_embed, text_pooler_out], axis=1
            )

            num_warmup_steps = 0

            with self.progress_bar(total=num_upscale_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    sigma = self.scheduler.sigmas[i]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        np.concatenate([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    scaled_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    scaled_model_input = np.concatenate(
                        [scaled_model_input, image_cond], axis=1
                    )
                    # preconditioning parameter based on  Karras et al. (2022) (table 1)
                    timestep = np.log(sigma) * 0.25

                    noise_pred = self.upscaler(
                        sample=scaled_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_condition,
                    ).sample

                    # in original repo, the output contains a variance channel that's not used
                    noise_pred = noise_pred[:, :-1]

                    # apply preconditioning, based on table 1 in Karras et al. (2022)
                    inv_sigma = 1 / (sigma**2 + 1)
                    noise_pred = (
                        inv_sigma * latent_model_input
                        + self.scheduler.scale_model_input(sigma, t) * noise_pred
                    )

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_output = self.scheduler.step(
                        noise_pred, t, torch.from_numpy(latents)
                    )
                    latents = scheduler_output.prev_sample.numpy()

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
        else:
            logger.debug("skipping latent upscaler, no model provided")

        # decode image
        latents = 1 / 0.18215 * latents

        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


def export_unet(pipeline, output_path, unet_sample_size=1024):
    device = torch.device("cpu")
    dtype = torch.float32

    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size

    unet_inputs = ["sample", "timestep", "encoder_hidden_states", "timestep_cond"]
    unet_in_channels = pipeline.unet.config.in_channels
    unet_path = output_path / "unet" / ONNX_MODEL

    logger.info("exporting UNet to %s", unet_path)
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(
                2,
                unet_in_channels,
                unet_sample_size // LATENT_FACTOR,
                unet_sample_size // LATENT_FACTOR,
            ).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            torch.randn(2, 64, 64, 2).to(
                device=device, dtype=dtype
            ),  # TODO: not the right shape
        ),
        output_path=unet_path,
        ordered_input_names=unet_inputs,
        # has to be different from "sample" for correct tracing
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch"},  # , 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        opset=14,
        half=False,
        external_data=True,
        v2=False,
    )


def load_and_export(output, source="stabilityai/sd-x2-latent-upscaler"):
    from pathlib import Path

    from diffusers import StableDiffusionLatentUpscalePipeline

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        source, torch_dtype=torch.float32
    )
    export_unet(upscaler, Path(output))


def load_and_run(
    prompt,
    output,
    source="stabilityai/sd-x2-latent-upscaler",
    checkpoint="../models/stable-diffusion-onnx-v1-5",
):
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        StableDiffusionLatentUpscalePipeline,
    )

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(source)
    highres = OnnxStableDiffusionHighresPipeline.from_pretrained(checkpoint)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        f"{checkpoint}/scheduler"
    )

    # combine them
    highres.scheduler = scheduler
    highres.upscaler = RetorchModel(upscaler.unet)

    # run
    result = highres.text2img(prompt, num_inference_steps=25, num_upscale_steps=25)
    image = result.images[0]
    image.save(output)


class RetorchModel:
    """
    Shim back from ONNX to PyTorch
    """

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, **kwargs):
        inputs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        outputs = self.model(**inputs)
        return UNet2DConditionOutput(sample=outputs.sample.numpy())
