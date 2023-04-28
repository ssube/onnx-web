from typing import Union
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution, DecoderOutput
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from ...server import ServerContext
from diffusers import OnnxRuntimeModel

logger = getLogger(__name__)

LATENT_CHANNELS = 4
SAMPLE_SIZE = 32

# TODO: does this need to change for fp16 modes?
timestep_dtype = np.float32


def set_vae_dtype(dtype):
    global timestep_dtype
    timestep_dtype = dtype


class VAEWrapper(object):
    def __init__(self, server: ServerContext, wrapped: OnnxRuntimeModel, decoder: bool):
        self.server = server
        self.wrapped = wrapped
        self.decoder = decoder

        self.tile_sample_min_size = SAMPLE_SIZE
        self.tile_latent_min_size = SAMPLE_SIZE
        self.tile_overlap_factor = 0.25

        self.quant_conv = nn.Conv2d(2 * LATENT_CHANNELS, 2 * LATENT_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)

    def __call__(self, latent_sample=None, **kwargs):
        global timestep_dtype

        logger.trace("VAE %s parameter types: %s", ("decoder" if self.decoder else "encoder"), latent_sample.dtype)
        if latent_sample.dtype != timestep_dtype:
            logger.info("converting VAE sample dtype")
            latent_sample = latent_sample.astype(timestep_dtype)

        if self.decoder:
            return self.tiled_decode(latent_sample, **kwargs)
        else:
            return self.tiled_encode(latent_sample, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def blend_v(self, a, b, blend_extent):
        for y in range(min(a.shape[2], b.shape[2], blend_extent)):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        for x in range(min(a.shape[3], b.shape[3], blend_extent)):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    @torch.no_grad()
    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is:
        different from non-tiled encoding due to each tile using a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            x (`torch.FloatTensor`): Input batch of images. return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`AutoencoderKLOutput`] instead of a plain tuple.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = torch.from_numpy(self.wrapped(latent_sample=tile.numpy()))
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        posterior = posterior.numpy()

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @torch.no_grad()
    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""Decode a batch of images using a tiled decoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled decoding is:
        different from non-tiled decoding due to each tile using a different decoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
            `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = torch.from_numpy(self.wrapped(latent_sample=tile.numpy()))
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        dec = dec.numpy()

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)