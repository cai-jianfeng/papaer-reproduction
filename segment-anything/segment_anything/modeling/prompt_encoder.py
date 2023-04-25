# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # nn.Embedding: 将 num_embeddings 数量的词典(即一个数)嵌入到长为 embedding_dim 的嵌入向量
        # 这里对于每个 point 都将它嵌入成长度为 embed_dim 的向量
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1, C=embed_dim x H x W

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)  # 为每张图片添加一个坐标为(0, 0)的 point
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)  # 类别为 -1
            points = torch.cat([points, padding_point], dim=1)  # points.shape = (batch_size, num_points+1, 2)
            labels = torch.cat([labels, padding_label], dim=1)  # labels.shape = (batch_size, num_points+1)
        # 将 point 的坐标归一化到 [0, 1], 再使用 随机高斯分布 embed 成 embed_dim // 2, 最后使用 cos, sin 对其操作并 concat, 最终变成 embed_dim
        # 此操作是对 point 的坐标进行位置编码
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)  # point_embedding.shape = (batch_size, num_points, embed_dim)
        # 将 (0, 0) 处的位置编码设为 0, 并加上 not_a_point_embed(表示它不是一个我们输入的 point)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        # 分别对 label = 0/1 的所有 point 加上 point_embeddings[0]/[1]
        point_embedding[labels == 0] += self.point_embeddings[0].weight  # label=0 表示 在 mask 之外; =1 表示在 mask 之内
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding  # point_embedding.shape = (batch_size, num_points, embed_dim)

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        # 猜测!: N = 1, 即每张图只有一个 box prompt
        # 猜测!: boxes.shape = (B, N, 2, 2) 表示 batch_size=B, 每张图有 N 个 box, 每个 box 有两个点表示: 左上+右下, 每个点有两个坐标值
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)  # coords.shape = (BxN, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)  # corner_embedding.shape = (BxN, 2, embed_dim)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # self.point_embeddings[2] 是 box 的左上角坐标的编码
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # self.point_embeddings[3] 是 box 的右下角坐标的编码
        return corner_embedding  # corner_embedding.shape = (BxN, 2, embed_dim)

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        # 猜测!: mask.shape = (B, 1, mask_h, mask_w)
        # h = mask_h / 4, w = mask_w / 4
        mask_embedding = self.mask_downscaling(masks)  # mask_embedding.shape = (B, embed_dim, h, w)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)  # 获取 prompt 的 batch size
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())  # shape = (2, 0, 3), 打印出来为一个空的tensor
        if points is not None:
            coords, labels = points  # coords.shape = (B, N, 2); labels.shape = (B, N)
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))  # point_embedding.shape = (batch_size, num_points, embed_dim)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)  # sparse_embeddings.shape = (batch_size, 0+num_points, embed_dim)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)  # box_embeddings.shape = (BxN, 2, embed_dim)
            # 由此可以推断, 每张图片只有一个 box, 即 N=1
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)  # sparse_embeddings.shape = (batch_size, 0+num_points+2, embed_dim)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)  # dense_embeddings.shape = (B, embed_dim, h, w)
        else:
            # weight.shape = (1, embed_dim, 1, 1) -> expand(bs, embed_dim, H, W)
            #   H, W 表示 image 打成 patch 块后的高宽
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings  # sparse_embeddings.shape = (batch_size, 0+num_points+2, embed_dim);
#                                                     dense_embeddings.shape = (B, embed_dim, h, w)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),  # 服从(0, 1)高斯分布的位置编码: shape=(2, embed_dim // 2)
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords.shape = (B, N, 2)
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1  # 将 point 的坐标范围扩大到 [-1, 1]
        # positional_encoding_gaussian_matrix.shape = (2, embed_dim // 2); 因为 positional_encoding_gaussian_matrix 的范围为 (0, 1)
        # 且 coords 的最后一个维度为 2, 即矩阵相乘时为 axb+cxd, 其中a, c in [-1, 1], b, d in [0, 1], 所以 axb+cxd in [-2, 2]
        coords = coords @ self.positional_encoding_gaussian_matrix  # coords.shape = (B, N, embed_dim // 2)
        coords = 2 * np.pi * coords  # coords in [-4pi, 4pi]
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # (B, N, embed_dim)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5  # cumsum: 累加
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h  # (h, w)
        x_embed = x_embed / w  # (h, w)
        # stack: (h, w, 2)
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C=embed_dim x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        # coords_input.shape = (batch_size, num_points, 2)
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 进行位置归一化: 归一化到(0,1)范围内
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
