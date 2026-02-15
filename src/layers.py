"""
StyleGAN2のビルディングブロック（レイヤー部品）

このモジュールには、StyleGAN2アーキテクチャで使用される
基本的なレイヤーとブロックが含まれています。
"""

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class EqualizedWeight(nn.Module):
    """学習率均等化のための重みラッパー
    
    Progressive Growingで導入された手法。重みを動的にスケーリングすることで、
    学習の安定性を向上させる。
    """

    def __init__(self, shape):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    """学習率均等化を適用した全結合層"""

    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """学習率均等化を適用した畳み込み層"""

    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class Conv2dWeightModulate(nn.Module):
    """スタイルベクトルによる重み変調を行う畳み込み層
    
    StyleGAN2のコア技術。スタイルベクトルで畳み込み重みを
    変調・復調することで、高品質な画像生成を実現。
    """

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class StyleBlock(nn.Module):
    """StyleGAN2のスタイルブロック
    
    スタイルベクトルとノイズを組み合わせて特徴マップを生成。
    """

    def __init__(self, w_dim, in_features, out_features):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToGray(nn.Module):
    """特徴マップをグレースケール画像に変換するレイヤー"""

    def __init__(self, w_dim, features):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, features, bias=1.0)
        self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class DiscriminatorBlock(nn.Module):
    """Discriminator用の残差ブロック"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale
