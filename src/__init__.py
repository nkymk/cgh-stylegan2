"""
CGH-StyleGAN: ホログラム生成のためのStyleGAN2実装

このパッケージは、計算機合成ホログラム(CGH)を生成するための
StyleGAN2ベースの深層学習モデルを提供します。

モジュール:
    - simulator: ホログラムシミュレータ（コア技術）
    - models: Generator, Discriminator, MappingNetwork
    - layers: StyleGAN2のビルディングブロック
    - dataset: データローダー
    - loss: 損失関数
"""

from .simulator import TorchHologramSimulator, simulate_hologram_batch
from .models import Generator, Discriminator, MappingNetwork
from .loss import PathLengthPenalty, gradient_penalty
from .dataset import get_loader

__all__ = [
    'TorchHologramSimulator',
    'simulate_hologram_batch',
    'Generator',
    'Discriminator',
    'MappingNetwork',
    'PathLengthPenalty',
    'gradient_penalty',
    'get_loader',
]
