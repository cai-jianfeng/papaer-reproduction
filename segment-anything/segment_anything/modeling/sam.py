# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)  # shape = (3, 1, 1)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # self.preprocess: 将 batched_input 中的所有 图片 x 同一 normalize 并 pad 成统一大小
        # input_images.shape = (batch_size, 3, image_encoder.img_size, image_encoder.img_size)
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        # shape=(batch_size, prompt_embed_dim, image_encoder.img_size//patch_size, image_encoder.img_size//patch_size)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )  # sparse_embeddings.shape = (1, 0+num_points+2, embed_dim);
            #    dense_embeddings.shape = (1, embed_dim, h, w)
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),  # shape = (1, C=embed_dim x H x W)
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )  # low_res_masks.shape = (1, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4);
            #    iou_predictions.shape = (1, num_mask_tokens)
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )  # masks.shape = (1, num_mask_tokens, original_size[0], original_size[1])
            masks = masks > self.mask_threshold  # 进行二值化处理, <mask_threshold 就表示不是 mask
            outputs.append(
                {
                    "masks": masks,  # masks.shape = (1, num_mask_tokens, original_size[0], original_size[1])
                    "iou_predictions": iou_predictions,  # iou_predictions.shape = (1, num_mask_tokens)
                    "low_res_logits": low_res_masks,
                    # low_res_masks.shape = (1, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4);
                    # 比原始的图像小 patch_size / 4 倍(通常 patch_size = 16), 所以是低分辨率下的 mask, 可以作为新的 mask 继续生成高质量的高分辨率 mask
                }
            )
        return outputs  # list[dict{str, Any}]

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        首先双线性插值到 image_encoder.img_size(vit-h=1024)的大小;
        然后取 input_size[0] x input_size[1] 的大小;
        最后, 再双线性插值到 original_size 的大小
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # masks.shape = (1, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # ... 表示前面所有的维度
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks  # masks.shape = (1, num_mask_tokens, original_size[0], original_size[1])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # x.shape = (3, H, W)
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x  # shape = (3, image_encoder.img_size, image_encoder.img_size)
