# Unconditional Hologram Generation with StyleGAN2

**StyleGAN2を用いた計算機合成ホログラフィの無条件生成**

## 概要

- 本リポジトリは、**StyleGAN2**を用いて計算機合成ホログラフィ（CGH）を生成するPyTorch実装です。
- 従来の深層学習ベースのCGH生成手法の多くは、2次元画像（透視画像や深度画像）を入力とする「条件付き生成」であり、3次元情報の欠落や近似が課題でした。
- 本研究では、ホログラムの干渉縞を一種のテクスチャとみなし、乱数から直接ホログラムを生成する**無条件生成モデル**を提案します。
- さらに、ホログラム画像だけでなく、物理シミュレーションによる**再生像の品質も同時に評価するMulti-Discriminator構成**を導入し、光学的に妥当な干渉縞の獲得を目指しました。

<img width="692" height="562" alt="image" src="https://github.com/user-attachments/assets/95058602-db83-4da9-b53e-10236398d0a8" />

## 特徴

* **無条件生成 (Unconditional Generation)**:
    * 特定の物体画像を入力とせず、潜在変数 $z$ (乱数) から多様なホログラムパターンを生成。
* **微分可能ホログラムシミュレータ (Differentiable Hologram Simulator)**:
    * PyTorch上でフレネル回折計算（FFTベース）を実装。
    * 生成器から再生像評価まで誤差逆伝播が可能。
* **Multi-Discriminator Architecture**:
    * **$D_{holo}$**: 生成された干渉縞パターンの統計的性質を評価。
    * **$D_{rec}$**: シミュレータを経て得られた再生像の光学的整合性を評価。

## 手法

### ネットワークアーキテクチャ
- 基盤モデルには高解像度画像生成に優れた **StyleGAN2** を採用しています。
- Generatorは $4 \times 4$ から段階的に解像度を上げ、 $512 \times 512$ の干渉縞画像を生成します。
- 学習は **WGAN-GP** の枠組みに基づき、以下の損失関数を最小化します。

$$
\mathcal{L}_{G} = -\lambda_{\mathrm{holo}} \cdot \mathbb{E}[D_{\mathrm{holo}}(G(z))] - \lambda_{\mathrm{rec}} \cdot \mathbb{E}[D_{\mathrm{rec}}(\mathrm{Sim}(G(z)))]
$$

ここで、 $\mathrm{Sim}(\cdot)$ は回折シミュレータを表します。

<img width="1322" height="482" alt="image" src="https://github.com/user-attachments/assets/abe4ea97-c1e7-4cd7-afe2-8868ada870ba" />

### データセット
高知大学 高田研究室により開発されたアルゴリズムを用い、15種類の3D点群データ（bunny, earth, chess等）から生成されたCGH画像を使用しています。

<img width="1313" height="592" alt="image" src="https://github.com/user-attachments/assets/99cc2c46-132f-4b1a-861d-323107ea962a" />

<img width="1293" height="632" alt="image" src="https://github.com/user-attachments/assets/7e3e3ac6-1870-4c7a-a458-445b28935f6b" />

## 実験条件

<img width="1006" height="540" alt="image" src="https://github.com/user-attachments/assets/7ccde90a-6d52-491b-aae1-ded5aa107afc" />


## 実験結果

### 生成画像の推移
学習が進むにつれて、ホログラム特有の高周波な干渉縞パターンが獲得されていることが確認できます。

<img width="1159" height="528" alt="image" src="https://github.com/user-attachments/assets/ba7dafaf-3eec-4308-a537-d9032791655a" />


### 再生像シミュレーション
生成されたホログラムに対して数値再生を行った結果です。

* **実験1 (Single Discriminator)**: 物体光と共役光が分離せず拡散
* **実験2 (Multi-Discriminator)**: 物体光の局在性が向上し、共役像との分離傾向

<img width="1291" height="599" alt="image" src="https://github.com/user-attachments/assets/c58b9c3c-8947-4369-8fbd-b59033dcca35" />


※ 現状では明瞭な結像には至っておらず、位相情報の欠落などが課題として挙げられます。

## 動作環境

* Python 3.8+
* PyTorch 1.10+
* NVIDIA GPU (Recommended: RTX 3090 x2 for training)
* CUDA Toolkit

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
    ├── simulator.py       # ホログラムシミュレータ
    ├── models.py          # Generator, Discriminator, MappingNetwork
    ├── layers.py          # StyleGAN2のビルディングブロック
    ├── dataset.py         # DataLoader
    └── loss.py            # 損失関数
```

## セットアップ

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
