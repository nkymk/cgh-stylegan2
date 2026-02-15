# CGH-StyleGAN

**ホログラム生成のためのStyleGAN2実装**

計算機合成ホログラム（CGH: Computer-Generated Hologram）を深層学習で生成するための研究プロジェクトです。StyleGAN2アーキテクチャをベースに、物理ベースのホログラムシミュレータを組み込むことで、高品質なホログラムパターンを生成します。

## 特徴

- **物理ベースシミュレータ**: Angular Spectrum法に基づく微分可能なホログラムシミュレータ
- **デュアルDiscriminator**: CGHパターンと再生像の両方を評価
- **Mixed Precision Training**: bf16/fp16による高速学習
- **マルチGPU対応**: Accelerateによる分散学習サポート

## ディレクトリ構成

```
cgh-stylegan/
├── README.md               # このファイル
├── requirements.txt        # 依存ライブラリ
├── .gitignore             # Git除外設定
├── config.py              # デフォルト設定値
├── train.py               # 学習スクリプト
├── generate.py            # 画像生成スクリプト
└── src/                   # ソースコード
    ├── __init__.py
    ├── simulator.py       # ホログラムシミュレータ（コア技術）
    ├── models.py          # Generator, Discriminator, MappingNetwork
    ├── layers.py          # StyleGAN2のビルディングブロック
    ├── dataset.py         # DataLoader
    └── loss.py            # 損失関数
```

## セットアップ

### 要件

- Python 3.8+
- CUDA 11.0+ (GPU学習の場合)
- PyTorch 2.0+

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-username/cgh-stylegan.git
cd cgh-stylegan

# 依存ライブラリのインストール
pip install -r requirements.txt
```

## 使い方

### 学習

```bash
# シングルGPU
python train.py --dataset_path /path/to/your/data

# マルチGPU (2GPU)
accelerate launch --num_processes=2 train.py --dataset_path /path/to/your/data

# カスタム設定
python train.py \
    --dataset_path /path/to/data \
    --output_dir ./outputs \
    --epochs 300 \
    --batch_size 6 \
    --lr 1e-4 \
    --mixed_precision bf16
```

### 画像生成

```bash
# 学習済みモデルから画像生成
python generate.py \
    --checkpoint_path ./checkpoints/model.pt \
    --output_dir ./generated \
    --num_samples 100
```

### 主なオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--dataset_path` | 学習データのパス | (必須) |
| `--output_dir` | 出力ディレクトリ | `./outputs` |
| `--epochs` | エポック数 | 300 |
| `--batch_size` | バッチサイズ (GPU当たり) | 6 |
| `--lr` | 学習率 | 1e-4 |
| `--log_resolution` | 解像度のlog2 (9=512x512) | 9 |
| `--mixed_precision` | Mixed Precision (no/fp16/bf16) | bf16 |
| `--propagation_distance` | 伝播距離 [m] | 0.5 |
| `--wave_length` | 波長 [m] | 486e-9 |
| `--pitch` | ピクセルピッチ [m] | 7.56e-6 |

全オプションは `python train.py --help` で確認できます。

## シミュレータの単体使用

ホログラムシミュレータは独立したモジュールとして使用可能です：

```python
import torch
from src.simulator import TorchHologramSimulator, simulate_hologram_batch

# シミュレータの初期化
simulator = TorchHologramSimulator(
    image_shape=512,
    distance=0.5,           # 伝播距離 [m]
    pitch=7.56e-6,          # ピクセルピッチ [m]
    wave_length=486e-9,     # 波長 [m]
)

# ホログラムから再生像をシミュレート
hologram = torch.randn(1, 1, 512, 512)  # (B, C, H, W)
reconstructed = simulate_hologram_batch(hologram, simulator)
```

## データセット形式

ImageFolderフォーマットを使用：

```
dataset/
└── train/
    └── class_name/
        ├── image001.bmp
        ├── image002.bmp
        └── ...
```

※ グレースケール画像を使用してください。

## 技術詳細

### ホログラムシミュレータ

Angular Spectrum法に基づく光波伝播シミュレーション：

1. **FFTベースの実装**: PyTorchのFFTを使用し、完全に微分可能
2. **自然サンプリング**: エイリアシングを防ぐため、適切なサイズにパディング
3. **fp32計算**: FFTの精度維持のため、常にfp32で計算

### デュアルDiscriminator構成

- **critic_holo**: CGHパターン（入力ホログラム）を評価
- **critic_rec**: 再生像（シミュレーション結果）を評価

両方のDiscriminatorからのフィードバックにより、物理的に妥当なホログラムパターンが生成されます。

## 引用

このコードを使用した場合は、以下を引用してください：

```bibtex
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## ライセンス

MIT License

## 謝辞

- [StyleGAN2](https://github.com/NVlabs/stylegan2) - NVIDIA
- [Accelerate](https://github.com/huggingface/accelerate) - Hugging Face
