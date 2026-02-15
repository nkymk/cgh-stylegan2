#!/usr/bin/env python
# coding: utf-8
"""
CGH-StyleGAN 画像生成スクリプト

学習済みモデルからホログラム画像を生成します。

使用方法:
    # 基本的な使い方
    python generate.py --checkpoint_path ./checkpoints/model.pt --num_samples 100
    
    # 出力先を指定
    python generate.py --checkpoint_path ./checkpoints/model.pt --output_dir ./generated
    
    # シードを固定して再現可能な生成
    python generate.py --checkpoint_path ./checkpoints/model.pt --seed 42
"""

import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src import Generator, MappingNetwork
import config


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='CGH-StyleGAN: 学習済みモデルからホログラム画像を生成'
    )
    
    # 必須引数
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='学習済みモデルのチェックポイントパス'
    )
    
    # 出力設定
    parser.add_argument(
        '--output_dir', type=str, default='./generated',
        help='出力ディレクトリ (default: ./generated)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='生成するサンプル数 (default: 100)'
    )
    parser.add_argument(
        '--format', type=str, default='bmp',
        choices=['bmp', 'png', 'jpg'],
        help='出力画像フォーマット (default: bmp)'
    )
    
    # モデル設定
    parser.add_argument(
        '--log_resolution', type=int, default=config.LOG_RESOLUTION,
        help=f'画像解像度のlog2 (default: {config.LOG_RESOLUTION})'
    )
    parser.add_argument(
        '--z_dim', type=int, default=config.Z_DIM,
        help=f'潜在空間の次元 (default: {config.Z_DIM})'
    )
    parser.add_argument(
        '--w_dim', type=int, default=config.W_DIM,
        help=f'中間潜在空間の次元 (default: {config.W_DIM})'
    )
    
    # その他
    parser.add_argument(
        '--seed', type=int, default=None,
        help='乱数シード（指定すると再現可能な生成）'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='使用デバイス (default: cuda)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='バッチサイズ（メモリに余裕があれば増やせる）'
    )
    
    return parser.parse_args()


def get_w(batch_size, device, mapping_net, w_dim, log_resolution):
    """潜在変数 w を生成する (形状: L, B, D)"""
    z = torch.randn(batch_size, w_dim, device=device)
    w = mapping_net(z)
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


def load_checkpoint(checkpoint_path, device, args):
    """チェックポイントを読み込む"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Generatorの初期化
    gen = Generator(args.log_resolution, args.w_dim)
    mapping_net = MappingNetwork(args.z_dim, args.w_dim)
    
    # 重みの読み込み
    if 'generator' in checkpoint:
        gen.load_state_dict(checkpoint['generator'])
    elif 'gen' in checkpoint:
        gen.load_state_dict(checkpoint['gen'])
    else:
        # チェックポイント全体がモデルの重みの場合
        gen.load_state_dict(checkpoint)
    
    if 'mapping_network' in checkpoint:
        mapping_net.load_state_dict(checkpoint['mapping_network'])
    elif 'mapping_net' in checkpoint:
        mapping_net.load_state_dict(checkpoint['mapping_net'])
    
    gen = gen.to(device)
    mapping_net = mapping_net.to(device)
    
    gen.eval()
    mapping_net.eval()
    
    return gen, mapping_net


def generate_images(gen, mapping_net, args):
    """画像を生成して保存"""
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} images...")
    print(f"Output directory: {args.output_dir}")
    print(f"Format: {args.format}")
    
    num_generated = 0
    
    with torch.no_grad():
        pbar = tqdm(total=args.num_samples)
        
        while num_generated < args.num_samples:
            # 残りの枚数を計算
            remaining = args.num_samples - num_generated
            current_batch = min(args.batch_size, remaining)
            
            # 生成
            w = get_w(current_batch, device, mapping_net, args.w_dim, args.log_resolution)
            noise = get_noise(current_batch, device, args.log_resolution)
            images = gen(w, noise)  # [-1, 1]
            
            # 保存
            for i in range(current_batch):
                img = images[i]
                
                # [0, 255] uint8 に変換
                img = (img.clamp(-1, 1) + 1) / 2  # → [0,1]
                img = img.mul(255).clamp(0, 255).byte()
                
                pil_img = to_pil_image(img.cpu())
                
                filename = f"img_{num_generated:05d}.{args.format}"
                filepath = os.path.join(args.output_dir, filename)
                pil_img.save(filepath)
                
                num_generated += 1
                pbar.update(1)
        
        pbar.close()
    
    print(f"Done! Generated {num_generated} images.")


def main():
    args = parse_args()
    
    # デバイス設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # シード設定
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # モデル読み込み
    gen, mapping_net = load_checkpoint(args.checkpoint_path, device, args)
    
    # 画像生成
    generate_images(gen, mapping_net, args)


if __name__ == "__main__":
    main()
