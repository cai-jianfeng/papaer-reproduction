# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            注意: use_abs_pos 和 use_rel_pos 是不同的,
                绝对位置编码是在ViT开头, 表示 patch_embedding 在原始图片中的位置;
                相对位置编码是在ViT中间, 每个 transformer block 中的 attantion map, 表示 q 和 k 在行/列中的差(q所在的行-k所在的行), 每个相同差使用相同的位置编码
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size  # vit-h = 1024, 表示宽和高, 即为一张正方形图

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            # 记录 image 的 各个 patch 块的 2维位置: 二维矩阵
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )  # batch 中的每个图片都使用相同的 pos_embed

        self.blocks = nn.ModuleList()  # transformer块
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),  # kernel_size = 1 -> 相当于只改变通道数
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),  # kernel_size = 3, padding = 1 -> 也不改变输入的 H, W
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (B, H, W, C)
        x = self.patch_embed(x)  # x.shape = (B, H//patch_size, W//patch_size, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed  # x.shape = (B, H//patch_size, W//patch_size, C)

        for blk in self.blocks:
            x = blk(x)  # x.shape = (B, H//patch_size, W//patch_size, C)
        # x.permute: -> x.shape = (B, C, H//patch_size, W//patch_size)
        x = self.neck(x.permute(0, 3, 1, 2))  # x.shape = (B, C, H//patch_size, W//patch_size)

        return x  # x.shape = (B, C, H//patch_size, W//patch_size)


class Block(nn.Module):
    # 与最原始的transformer不同, 加入了 window attention
    # 同时, 它是先 norm 再 attetnion/FFN, 不同于原始 transformer 的先 attention/FFN 再 norm
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)  # nn.LayerNorm
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size  # 14 for vit-h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x,
                                         self.window_size)  # x.shape = [B * num_windows, window_size, window_size, C]

        x = self.attn(x)  # x.shape = [B * num_windows, window_size, window_size, C]
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))  # x.shape = [B, H, W, C]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # x.shape = [B, H, W, C]

        return x  # x.shape = [B, H, W, C]


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 1 / head_dim^0.5

        self.qkv = nn.Linear(dim, dim * 3,
                             bias=qkv_bias)  # transformer需要对 q, k, v进行 embed, 这里使用 Linear(dim, dim*3) 实现一次性将 q, k, v 分别进行 embed
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:  # 使用相对位置编码
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            # rel_pos.shape = (2 * window_size/input_size - 1 ???, head_dim)
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape  # x此时已经被 patch_embedding, shape = (B, input_size // patch_size, input_size // patch_size, patch_embed_dim)
        # qkv with shape (3, B, nHead, H * W, C)
        # self.qkv(x) -> (B, H, W, dim x 3); reshape -> (B, HxW, 3, num_heads, dim / num_heads=C, 即head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # unbind: 移除指定维后, 返回一个元组, 包含了沿着指定维切片后的各个切片
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # (HxW, HxW) 中的 (i,j): 表示第 i 个 q 和 第 j 个 k 的点乘结果 -> 所以每一行(i)是第 i 个 q 和所有 k 的点乘结果
        attn = (q * self.scale) @ k.transpose(-2, -1)  # @ 表示矩阵乘法 -> attn.shape = (B * nHead, H * W, H * W)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W),
                                          (H, W))  # attn.shape = (B * nHead, q_h * q_w, k_h * k_w)

        attn = attn.softmax(dim=-1)  # 在最后一个维度进行 softmax: 表示 BxnHead x q_hxq_w 个的每一个 query 在 k_hxk_w 个 k 上的相似度概率
        # H, W = q_h, q_w = k_h, k_w
        # attn @ v: shape = (B * nHead, q_h * q_w, k_h * k_w) @ (B * nHead, H * W, C) =(B * nHead, q_h * q_w, C=head_dim)
        #   将得到的 k_hxk_w 个 k 的各个相似概率, 再将概率与对应的 v 相乘并加和, 得到更新的 query
        # view: (B, nHead, q_h, q_w, C=head_dim) -> permute: (B, q_h, q_w, nHead, C=head_dim) —> reshape: (B, q_h, q_w, nHead * C = dim)
        # 将多头注意力的结果进行 concate, 即变成 (B, q_h, q_w, nHead * C = dim)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # 再对多头注意力的结果进行embedding
        x = self.proj(x)  # (B, q_h, q_w, nHead * C = dim)

        return x  # (B, q_h, q_w, nHead * C = dim)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    # H % window_size 表示 H 整除 window_size 后的余数, 即该部分不是一个完整的 window_size, 需要 padding
    # window_size - H % window_size 表示 需要 padding 的个数
    # % window_size 避免整除时的特殊情况: 整除时 H % window_size = 0, 理论上不需要 padding, 但是 window_size - H % window_size = window_size,
    #                                 再 % window_size 就等于 0
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        # F.pad 的 pad 参数成对出现, 表示对于每个维度的左边和右边的 padding 个数
        # 需要注意: F.pad 时是按照 B, W, H, C 的顺序, 而 x 是 B, H, W, C 的顺序, 所以 W 和 H pad 时需要交换位置
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # permute: (B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size,
                                                            C)  # shape = [B * num_windows, window_size, window_size, C]
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (
                Hp * Wp // window_size // window_size)  # (Hp * Wp // window_size // window_size) = num_windows
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x  # x.shape = (B, H, W, C)


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    2维的相对位置编码
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos.shape = (2 * window_size/input_size - 1, head_dim)
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        # F.interpolate: 上/下采样 -> size 表示采样后的shape, 如果只有一个, 则表示最后一个维度的采样后的维度值, 其他不变
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),  # rel_pos.shape = (1, C, L)
            size=max_rel_dist,
            mode="linear",
        )  # rel_pos_resized.shape = (1, C, max_rel_dist)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)  # shape = (max_rel_dist, C)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    # 使用较短的长度缩放坐标, 即最终将坐标缩放到较长的长度 -> 下面代码将 q_coords, k_coords 缩放到 (0, .. , max(q_size, k_size)-1)
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)  # q_coords.shape = (q_size, 1)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)  # k_coords.shape = (1, k_size)
    # (q_coords - k_coords).shape = (q_size, k_size): (i, j)表示第 i 行/列的 q 与 第 j 行/列的 k 在 行/列 坐标上相差多少(i-j)
    # + (k_size - 1) * max(q_size / k_size, 1.0): 保证相对位置坐标是从 0 开始, 没有负数
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    # 对于relative_coords(shape=(q_size, k_size)) 中的 (i, j), 在 rel_pos-resized(shape=(max_rel_dist, C)) 中取 relative_coords[i][j] 处的元素
    # 所以相对位置相差为 x 的 q 和 k 的相对位置编码相同
    return rel_pos_resized[relative_coords.long()]  # (q_size, k_size, C)


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos.shape = (2 * window_size/input_size - 1, head_dim)
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    # 相对位置编码: 相对于行/列, q所在的行/列相对于k所在的行/列的差:
    #   当 q 的行/列 = 0, k 的行/列 = k_size-1 时, 定义行/列差为 0, 则位置编码 = rel_pos_h[0]/rel_pos_w[0], 从而依次递增;
    #   当 q 的行/列 = q_size-1, k 的行/列 = 0 时, 定义行/列差为 k_size + q_size - 2, 则位置编码 = rel_pos_h[k_size + q_size - 2]/rel_pos_w[k_size + q_size - 2]
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # (q_h, k_h, C)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # (q_w, k_w, C)

    B, _, dim = q.shape  # q.shape = (B, HxW, dim)
    r_q = q.reshape(B, q_h, q_w, dim)
    # rel_h 是 bhwc x hkc -> bhwk; 即 hwc x hkc -> hwk, 其中的 (i, j, k) 表示第 i 行, 第 j 列的 q 与第 i 行所有列的Rh(列的相对位置编码)的矩阵乘积结果
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)  # (B, q_h, q_w, k_h)
    # bhwc, wkc -> bhwk:
    #   bhwc permute bwhc
    #   wkc permute wck
    #   bwhc @ wck = bwhk
    #   bwhk permute bhwk
    # rel_w 同理, 不过变成了列
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)  # (B, q_h, q_w, k_w)
    # rel_h, rel_w 将每个相对位置编码转化为一个数, 而不是一个向量
    # rel_h 表示 q 和 k 在列上的相对位置编码, 与行无关, 所以每个 k_w 加的值都相同; 同理 rel_w
    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn  # attn.shape = (B, q_h * q_w, k_h * k_w)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    # 使用卷积神经网络实现将 image 打成 patch 并进行 embedding -> 将 kernel size 和 stride 设置成一样, padding = 0
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (B, 3, input_size, input_size)
        x = self.proj(x)  # x.shape = (B, embed_dim, input_size // patch_size, input_size // patch_size)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x  # x.shape = (B, input_size // patch_size, input_size // patch_size, embed_dim)
