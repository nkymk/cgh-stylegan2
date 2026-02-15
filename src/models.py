"""
StyleGAN2のネットワークモデル

このモジュールには、Generator, Discriminator, MappingNetworkの
メインネットワーク構造が定義されています。
"""

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

from .layers import (
    EqualizedLinear,
    EqualizedConv2d,
    StyleBlock,
    ToGray,
    DiscriminatorBlock,
)


class MappingNetwork(nn.Module):
    """潜在空間zから中間潜在空間wへのマッピングネットワーク
    
    8層のMLPで構成され、入力にPixelNormを適用。
    """

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim)
        )

    def forward(self, x):
        # PixelNorm
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(x)


class GeneratorBlock(nn.Module):
    """Generatorの1ブロック（2つのStyleBlockとToGray）"""

    def __init__(self, w_dim, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(w_dim, in_features, out_features)
        self.style_block2 = StyleBlock(w_dim, out_features, out_features)
        self.to_gray = ToGray(w_dim, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        gray = self.to_gray(x, w)
        return x, gray


class Generator(nn.Module):
    """StyleGAN2 Generator
    
    学習可能な定数から始まり、Progressive Growingと
    スキップ接続を使用して画像を生成。
    
    Args:
        log_resolution: 出力解像度のlog2 (例: 9 for 512x512)
        w_dim: 中間潜在空間の次元
        n_features: 基本特徴量数
        max_features: 最大特徴量数
    """

    def __init__(self, log_resolution, w_dim, n_features=32, max_features=256):
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(w_dim, features[0], features[0])
        self.to_gray = ToGray(w_dim, features[0])

        blocks = [GeneratorBlock(w_dim, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):
        """
        Args:
            w: スタイルベクトル (L, B, D) - レイヤー数 x バッチサイズ x 次元
            input_noise: 各レイヤー用のノイズリスト
        
        Returns:
            生成画像 (B, 1, H, W) - [-1, 1]の範囲
        """
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        gray = self.to_gray(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, gray_new = self.blocks[i - 1](x, w[i], input_noise[i])
            gray = F.interpolate(gray, scale_factor=2, mode="bilinear") + gray_new

        return torch.tanh(gray)


class Discriminator(nn.Module):
    """StyleGAN2 Discriminator
    
    Minibatch Standard Deviationを使用してmode collapseを防止。
    
    Args:
        log_resolution: 入力解像度のlog2 (例: 9 for 512x512)
        n_features: 基本特徴量数
        max_features: 最大特徴量数
        mbstd_group_size: Minibatch Stdのグループサイズ
            4  = StyleGAN2標準（局所的な多様性を見る）
            -1 = バッチサイズ全体を使用（大域的な多様性を見る）
    """

    def __init__(self, log_resolution, n_features=64, max_features=256, mbstd_group_size=4):
        super().__init__()
        self.mbstd_group_size = mbstd_group_size

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_gray = nn.Sequential(
            EqualizedConv2d(1, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        """Minibatch Standard Deviation
        
        StyleGAN2公式準拠の実装。バッチ内の統計情報を
        追加チャンネルとして連結。
        """
        N, C, H, W = x.shape
        
        # グループサイズの決定
        if self.mbstd_group_size <= 0 or self.mbstd_group_size > N:
            G = N
        else:
            G = self.mbstd_group_size if N % self.mbstd_group_size == 0 else N
        
        M = N // G  # グループ数
        
        # [M, G, C, H, W] にreshape
        y = x.reshape(M, G, C, H, W)
        
        # グループ内(dim=1)で分散を計算
        variance = y.var(dim=1, unbiased=False)  # [M, C, H, W]
        std = torch.sqrt(variance + 1e-8)
        
        # 全チャンネル・全ピクセルの平均
        mean_std = std.mean(dim=[1, 2, 3], keepdim=True)  # [M, 1, 1, 1]
        
        # 元のバッチサイズに展開
        mean_std = mean_std.repeat(1, G, 1, 1).reshape(N, 1, 1, 1)  # [N, 1, 1, 1]
        batch_statistics = mean_std.repeat(1, 1, H, W)  # [N, 1, H, W]
        
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):
        x = self.from_gray(x)
        x = self.blocks(x)
        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)
