from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

logger = getLogger(__name__)


class UNet2DConditionModel_CNet(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        down_block_add_res00: Optional[torch.Tensor] = None,
        down_block_add_res01: Optional[torch.Tensor] = None,
        down_block_add_res02: Optional[torch.Tensor] = None,
        down_block_add_res03: Optional[torch.Tensor] = None,
        down_block_add_res04: Optional[torch.Tensor] = None,
        down_block_add_res05: Optional[torch.Tensor] = None,
        down_block_add_res06: Optional[torch.Tensor] = None,
        down_block_add_res07: Optional[torch.Tensor] = None,
        down_block_add_res08: Optional[torch.Tensor] = None,
        down_block_add_res09: Optional[torch.Tensor] = None,
        down_block_add_res10: Optional[torch.Tensor] = None,
        down_block_add_res11: Optional[torch.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        down_block_add_res = (
            down_block_add_res00,
            down_block_add_res01,
            down_block_add_res02,
            down_block_add_res03,
            down_block_add_res04,
            down_block_add_res05,
            down_block_add_res06,
            down_block_add_res07,
            down_block_add_res08,
            down_block_add_res09,
            down_block_add_res10,
            down_block_add_res11,
        )
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_add_res,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=return_dict,
        )
