"""
データセットとDataLoader

このモジュールには、学習に使用するデータローダーの
作成関数が含まれています。
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loader(
    dataset_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 2,
):
    """最適化されたDataLoaderを作成
    
    Args:
        dataset_path: データセットのルートパス
        image_size: 出力画像サイズ (正方形)
        batch_size: バッチサイズ
        num_workers: DataLoaderのワーカー数
        prefetch_factor: プリフェッチするバッチ数
    
    Returns:
        DataLoader: 学習用DataLoader
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,              # GPUへの転送を高速化
        persistent_workers=True,       # エポックごとの再初期化を防ぐ
        prefetch_factor=prefetch_factor,
    )
    
    return loader
