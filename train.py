#!/usr/bin/env python
# coding: utf-8
"""
CGH-StyleGAN 学習スクリプト

Mixed Precision Training + DDP対応のホログラム生成StyleGAN2

実行方法:
    # シングルGPU
    python train.py --dataset_path /path/to/data
    
    # マルチGPU (DDP)
    accelerate launch --num_processes=2 train.py --dataset_path /path/to/data
"""

import argparse
import os

import torch
from torch import optim
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from accelerate import Accelerator

from src import (
    TorchHologramSimulator,
    simulate_hologram_batch,
    Generator,
    Discriminator,
    MappingNetwork,
    PathLengthPenalty,
    gradient_penalty,
    get_loader,
)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='CGH-StyleGAN: ホログラム生成のためのStyleGAN2学習'
    )
    
    # 必須引数
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='学習データセットのパス'
    )
    
    # 出力設定
    parser.add_argument(
        '--output_dir', type=str, default='./outputs',
        help='出力ディレクトリ (default: ./outputs)'
    )
    
    # 学習設定
    parser.add_argument('--epochs', type=int, default=300, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=6, help='バッチサイズ (GPU1枚あたり)')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--log_resolution', type=int, default=9, help='画像解像度のlog2 (9=512x512)')
    
    # モデル設定
    parser.add_argument('--z_dim', type=int, default=256, help='潜在空間の次元')
    parser.add_argument('--w_dim', type=int, default=256, help='中間潜在空間の次元')
    parser.add_argument('--mbstd_group_size', type=int, default=-1, help='Minibatch Stdのグループサイズ')
    
    # ホログラム設定
    parser.add_argument('--propagation_distance', type=float, default=0.5, help='伝播距離 [m]')
    parser.add_argument('--wave_length', type=float, default=486e-9, help='波長 [m]')
    parser.add_argument('--pitch', type=float, default=7.56e-6, help='ピクセルピッチ [m]')
    parser.add_argument(
        '--crop_region', type=int, nargs=4, default=[1100, 1612, 3100, 3612],
        help='ホログラムのクロップ領域 (y_start y_end x_start x_end)'
    )
    
    # 損失関数の重み
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Gradient Penaltyの重み')
    parser.add_argument('--lambda_holo', type=float, default=0.1, help='CGHパターンのロス重み')
    parser.add_argument('--lambda_rec', type=float, default=1.0, help='再生像のロス重み')
    
    # データローダー設定
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoaderのワーカー数')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='プリフェッチファクター')
    
    # Mixed Precision設定
    parser.add_argument(
        '--mixed_precision', type=str, default='bf16',
        choices=['no', 'fp16', 'bf16'],
        help='Mixed Precision設定 (no/fp16/bf16)'
    )
    
    # ログ設定
    parser.add_argument('--log_interval', type=int, default=10, help='ログ出力間隔 (バッチ数)')
    parser.add_argument('--save_interval', type=int, default=50, help='サンプル保存間隔 (エポック数)')
    parser.add_argument('--num_samples', type=int, default=100, help='保存するサンプル数')
    
    return parser.parse_args()


def get_w(batch_size, device, mapping_net, w_dim, log_resolution):
    """潜在変数 w を生成する (形状: L, B, D)"""
    z = torch.randn(batch_size, w_dim, device=device)
    w = mapping_net(z)
    # Generatorは (Layers, Batch, Dim) を期待
    return w[None, :, :].expand(log_resolution, -1, -1)


def get_noise(batch_size, device, log_resolution):
    """各レイヤー用のノイズを生成する"""
    noise = []
    resolution = 4

    for i in range(log_resolution):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))
        resolution *= 2

    return noise


def generate_examples(gen, mapping_net, device, epoch, args):
    """サンプル画像を生成して保存"""
    gen.eval()
    
    for i in range(args.num_samples):
        with torch.no_grad():
            w = get_w(1, device, mapping_net, args.w_dim, args.log_resolution)
            noise = get_noise(1, device, args.log_resolution)
            img = gen(w, noise)  # [-1, 1] の1チャンネル画像

            # 画像を [0, 255] uint8 に変換
            img = (img.squeeze(0).clamp(-1, 1) + 1) / 2  # → [0,1]
            img = img.mul(255).clamp(0, 255).byte()      # → [0,255] に変換

            pil_img = to_pil_image(img.cpu())

            # 保存先作成
            save_dir = os.path.join(args.output_dir, f"epoch{epoch}")
            os.makedirs(save_dir, exist_ok=True)

            # BMP形式で保存
            pil_img.save(os.path.join(save_dir, f"img_{i}.bmp"))

    gen.train()


def train_fn(
    critic_holo,
    critic_rec,
    gen,
    mapping_network,
    path_length_penalty,
    loader,
    opt_critic_holo,
    opt_critic_rec,
    opt_gen,
    opt_mapping_network,
    simulator,
    accelerator,
    args,
):
    """DDP対応のトレーニング関数（Mixed Precision対応）"""
    loop = tqdm(loader, disable=not accelerator.is_local_main_process, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size, accelerator.device, mapping_network, args.w_dim, args.log_resolution)
        noise = get_noise(cur_batch_size, accelerator.device, args.log_resolution)
        
        fake = gen(w, noise)

        # ==============================================================================
        #  Discriminator (Critic) の学習
        # ==============================================================================
        
        critic_fake_holo = critic_holo(fake.detach())
        critic_real_holo = critic_holo(real)

        with torch.no_grad():
            real_sim = simulate_hologram_batch(real, simulator, normalize=True)
        
        fake_sim = simulate_hologram_batch(fake, simulator, normalize=True)

        critic_fake_rec = critic_rec(fake_sim.detach())
        critic_real_rec = critic_rec(real_sim)

        # Gradient Penalty（fp32で計算）
        gp_holo = gradient_penalty(critic_holo, real, fake, device=accelerator.device)
        gp_rec = gradient_penalty(critic_rec, real_sim, fake_sim, device=accelerator.device)

        loss_critic_holo = (
            -(torch.mean(critic_real_holo.float()) - torch.mean(critic_fake_holo.float()))
            + args.lambda_gp * gp_holo
            + (0.001 * torch.mean(critic_real_holo.float() ** 2))
        )
        loss_critic_rec = (
            -(torch.mean(critic_real_rec.float()) - torch.mean(critic_fake_rec.float()))
            + args.lambda_gp * gp_rec
            + (0.001 * torch.mean(critic_real_rec.float() ** 2))
        )

        total_critic_loss = loss_critic_holo + loss_critic_rec

        opt_critic_holo.zero_grad(set_to_none=True)
        opt_critic_rec.zero_grad(set_to_none=True)
        accelerator.backward(total_critic_loss)
        opt_critic_holo.step()
        opt_critic_rec.step()

        # ==============================================================================
        #  Generator の学習
        # ==============================================================================
        
        gen_fake_holo = critic_holo(fake)
        gen_fake_rec = critic_rec(fake_sim)
        
        loss_gen = -args.lambda_holo * torch.mean(gen_fake_holo) - args.lambda_rec * torch.mean(gen_fake_rec)

        # Path Length Penalty (頻度を落として計算)
        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        opt_mapping_network.zero_grad(set_to_none=True)
        opt_gen.zero_grad(set_to_none=True)
        accelerator.backward(loss_gen)
        opt_gen.step()
        opt_mapping_network.step()

        # ログ更新
        if batch_idx % args.log_interval == 0:
            loop.set_postfix(
                gp=gp_holo.item(),
                d_loss=total_critic_loss.item(),
                g_loss=loss_gen.item(),
            )


def main():
    args = parse_args()
    
    # Acceleratorの初期化
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Device: {accelerator.device}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")

    # DataLoader
    image_size = 2 ** args.log_resolution
    loader = get_loader(
        dataset_path=args.dataset_path,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Simulator
    crop_region = tuple(args.crop_region) if args.crop_region else None
    simulator = TorchHologramSimulator(
        image_shape=image_size,
        distance=args.propagation_distance,
        pitch=args.pitch,
        wave_length=args.wave_length,
        pad_to_natural=True,
        normalize=False,
        crop_region=crop_region,
    ).to(accelerator.device)

    # モデル
    gen = Generator(args.log_resolution, args.w_dim)
    critic_holo = Discriminator(args.log_resolution, mbstd_group_size=args.mbstd_group_size)
    critic_rec = Discriminator(args.log_resolution, mbstd_group_size=args.mbstd_group_size)
    mapping_network = MappingNetwork(args.z_dim, args.w_dim)
    path_length_penalty = PathLengthPenalty(0.99).to(accelerator.device)

    # Optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_critic_holo = optim.Adam(critic_holo.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_critic_rec = optim.Adam(critic_rec.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=args.lr, betas=(0.0, 0.99))

    # Acceleratorで準備
    (
        gen, critic_holo, critic_rec, mapping_network,
        opt_gen, opt_critic_holo, opt_critic_rec, opt_mapping_network,
        loader
    ) = accelerator.prepare(
        gen, critic_holo, critic_rec, mapping_network,
        opt_gen, opt_critic_holo, opt_critic_rec, opt_mapping_network,
        loader
    )

    gen.train()
    critic_holo.train()
    critic_rec.train()
    mapping_network.train()

    print(f"Training on {accelerator.num_processes} GPU(s)")
    print(f"Simulator target shape: {simulator.target_shape}")
    print(f"Image size: {image_size}x{image_size}")

    # 学習ループ
    for epoch in range(args.epochs):
        train_fn(
            critic_holo,
            critic_rec,
            gen,
            mapping_network,
            path_length_penalty,
            loader,
            opt_critic_holo,
            opt_critic_rec,
            opt_gen,
            opt_mapping_network,
            simulator,
            accelerator,
            args,
        )
        
        # サンプル保存
        if epoch % args.save_interval == 0 and accelerator.is_local_main_process:
            generate_examples(
                accelerator.unwrap_model(gen),
                accelerator.unwrap_model(mapping_network),
                accelerator.device,
                epoch,
                args,
            )
            print(f"Epoch {epoch}: Saved {args.num_samples} samples to {args.output_dir}/epoch{epoch}/")


if __name__ == "__main__":
    main()
