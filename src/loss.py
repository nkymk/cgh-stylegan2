"""
損失関数

このモジュールには、StyleGAN2の学習に使用される
損失関数とペナルティ項が含まれています。
"""

import torch
from torch import nn
from math import sqrt


class PathLengthPenalty(nn.Module):
    """Path Length Regularization
    
    潜在空間での移動が画像空間での変化と
    比例するように正則化。これにより、より
    滑らかで解釈可能な潜在空間が得られる。
    
    Args:
        beta: 指数移動平均の減衰率
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        """
        Args:
            w: スタイルベクトル
            x: 生成画像
        
        Returns:
            Path Length Penaltyの値
        """
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True
        )

        # 数値安定性のためにepsilonを追加
        norm = torch.sqrt((gradients ** 2).sum(dim=2).mean(dim=1) + 1e-8)

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss


def gradient_penalty(critic, real, fake, device="cpu"):
    """WGAN-GP Gradient Penalty（fp32固定）
    
    WGAN-GPでは勾配のノルムが重要なため、fp32で計算する必要がある。
    DDPでラップされたモデルでも正しく動作するよう、
    autocastを完全に無効化してから計算を行う。
    
    Args:
        critic: Discriminatorモデル
        real: 本物の画像バッチ
        fake: 生成画像バッチ
        device: 計算デバイス
    
    Returns:
        Gradient Penaltyの値
    """
    # fp16入力をfp32に変換（detachして計算グラフを切断）
    real = real.detach().float()
    fake = fake.detach().float()
    
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1), device=device, dtype=torch.float32)
    beta = beta.expand(BATCH_SIZE, C, H, W)
    
    interpolated_images = real * beta + fake * (1 - beta)
    interpolated_images.requires_grad_(True)

    # autocastを完全に無効化してcriticを実行
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # DDPラップを外して直接モデルにアクセス
        critic_module = critic.module if hasattr(critic, 'module') else critic
        mixed_scores = critic_module(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    # 数値安定性のためにeps追加
    gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)
    gp = torch.mean((gradient_norm - 1) ** 2)
    
    return gp
