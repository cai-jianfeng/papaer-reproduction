# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(  # 上采样
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),  # 反卷积
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                # input_dim, hidden_dim, output_dim
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)  # num_multimask_outputs + 1 个 MLP
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # image_embeddings.shape = (B, C=transformer_dim, H//patch_size, W//patch_size)
        masks, iou_pred = self.predict_masks(  # masks.shape = (batch_size, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4);
            #                                    iou_pred.shape = (batch_size, num_mask_tokens)
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )  # masks.shape = (batch_size, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4); iou_pred.shape = (batch_size, num_mask_tokens)

        # Select the correct mask or masks for output
        # slice: 有3个参数: start, stop, step -> 表示从 start 开始, 以 step 为步长, 到 stop 结束选择元素
        if multimask_output:
            mask_slice = slice(1, None)  # 去掉第一个 mask, 因为那是我们自己比原始的 num_mask 手动增加了 1 个
        else:
            mask_slice = slice(0, 1)  # 只保留第一个 mask, 我们希望这个我们自己手动添加的 mask 总结之后的所以 mask, 凝练出一个最好的 mask
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred  # masks.shape = (batch_size, num_mask_tokens, h=H//patch_sizex4, w=W//patch_sizex4);
        #                         iou_pred.shape = (batch_size, num_mask_tokens)

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image_embeddings.shape = (B=1, C=transformer_dim, H//patch_size, W//patch_size)
        # sparse_embeddings.shape = (batch_size, 0+num_points+2, embed_dim=transformer_dim);
        # dense_embeddings.shape = (B, embed_dim=transformer_dim, h, w)
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight],
                                  dim=0)  # output_tokens.shape = (1 + num_multimask_outputs+1, embed_dim)
        # unsqueeze: (1, num_multimask_outputs + 2, embed_dim) -> expand: (batch_size, 1 + num_multimask_outputs+1, embed_dim)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens.shape = (batch_size, num_multimask_outputs+2 + 0+num_points+2, embed_dim)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # torch.repeat_interleave: 将 tensor 沿特定维度重复
        # src.shape = (batch_size, C, H//patch_size, W//patch_size) -> H//patch_size = h, W//patch_size = w
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        # pos_src.shape = (batch_size, C, H//patch_size, W//patch_size)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # 将 iou_token, mask_tokens 放在 sparse_prompt_embeddings 之间组成 tokens 一起送入 transformer 与 image_embeddings 一起学习,
        # 希望将 sparse_prompt_embeddings 与 image_embeddings 的相关关系学习凝练到 iou_token, mask_tokens 中
        hs, src = self.transformer(src, pos_src,
                                   tokens)  # hs 是 point_embedding, shape=(batch_size, num_multimask_outputs+2 + 0+num_points+2, embed_dim);
        #                                                   src 是 image_embedding, shape=batch_size x N_image_tokens x C
        iou_token_out = hs[:, 0, :]  # (batch_size, 1, embed_dim)
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # (batch_size, num_multimask_outputs+1, embed_dim);

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # transpose: shape=batch_size x C x N_image_tokens -> view: shape=batch_size x C x h x w
        upscaled_embedding = self.output_upscaling(src)  # shape = batch_size x C//8 x h*4 x w*4
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # mask_tokens_out[:, i, :].shape = (batch_size, embed_dim)
            # hyper_in_list.shape = (num_mask_tokens, batch_size, embed_dim // 8)
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # hyper_in.shape = (batch_size, num_mask_tokens, embed_dim // 8)
        # 对于每个 mask_tokens, 都通过与 upscaled_embedding 的矩阵乘法获得 h * w 大小的掩码信息,
        # 所以我们是希望每个 mask_tokens 都已经学习到了一整个图的掩码信息, 根据我们提供的 sparse_prompt_embeddings 和 dense_prompt_embeddings
        # (每个mask_tokens学习不太的掩码信息, 比如 1 学习整个人, 2 学习衣服等)
        b, c, h, w = upscaled_embedding.shape
        # hyper_in @ upscaled_embedding.view(b, c, h * w).shape = (batch_size, num_mask_tokens, h * w)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h,
                                                                       w)  # (batch_size, num_mask_tokens, h, w)

        # Generate mask quality predictions
        # iou_token_out.shape = (batch_size, embed_dim)
        iou_pred = self.iou_prediction_head(iou_token_out)  # iou_pred.shape = (batch_size, num_mask_tokens)

        return masks, iou_pred  # masks.shape = (batch_size, num_mask_tokens, h, w); iou_pred.shape = (batch_size, num_mask_tokens)
        # h = H // patch_size x 4; w = W // patch_size x 4


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            # [input_dim] + h = [input_dim, hidden_dim, ..., hidden_dim]
            # h + [output_dim] = [hidden_dim, ..., hidden_dim, output_dim]
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
