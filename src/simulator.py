"""
ホログラムシミュレータ

このモジュールは、この研究のコア技術であるホログラムシミュレーションを実装しています。
Angular Spectrum法に基づく光波伝播シミュレーションをPyTorchで実装し、
Generatorの学習に使用します。

使用例:
    from src.simulator import TorchHologramSimulator, simulate_hologram_batch
    
    simulator = TorchHologramSimulator(
        image_shape=512,
        distance=0.5,
        pitch=7.56e-6,
        wave_length=486e-9,
    )
    reconstructed = simulate_hologram_batch(holograms, simulator)
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# デフォルトのホログラムパラメータ
DEFAULT_PITCH = 7.56e-6        # ピクセルピッチ [m]
DEFAULT_WAVE_LENGTH = 486e-9   # 波長 [m]


def _build_zone_plate_fft(
    image_shape,
    distance,
    pitch=DEFAULT_PITCH,
    wave_length=DEFAULT_WAVE_LENGTH,
    device=None,
    dtype=torch.float32
):
    """フレネルゾーンプレートのFFTを計算
    
    Args:
        image_shape: 出力画像サイズ (int または tuple)
        distance: 伝播距離 [m]
        pitch: ピクセルピッチ [m]
        wave_length: 光の波長 [m]
        device: 計算デバイス
        dtype: データ型
    
    Returns:
        ゾーンプレートのFFT (複素テンソル)
    """
    if isinstance(image_shape, int):
        height = width = image_shape
    else:
        height, width = image_shape
    if device is None:
        device = torch.device("cpu")
    
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    
    x = (xx - width // 2) * pitch
    y = (yy - height // 2) * pitch
    
    phase = (x ** 2 + y ** 2) * math.pi / (wave_length * distance)
    zone_plate = torch.polar(torch.ones_like(phase), phase)
    
    return torch.fft.fft2(zone_plate)


def _natural_size(distance, pitch=DEFAULT_PITCH, wave_length=DEFAULT_WAVE_LENGTH):
    """自然サンプリングサイズを計算
    
    Angular Spectrum法で正確なシミュレーションを行うために
    必要な最小ピクセル数を計算。
    """
    return int(math.ceil(wave_length * distance / (pitch ** 2)))


def _pad_tensor(tensor, size):
    """テンソルを指定サイズにゼロパディング"""
    if isinstance(size, int):
        target_h = target_w = size
    else:
        target_h, target_w = size
    _, _, h, w = tensor.shape
    if h == target_h and w == target_w:
        return tensor
    
    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))


def _center_crop(tensor, size):
    """テンソルを中央からクロップ"""
    if isinstance(size, int):
        target_h = target_w = size
    else:
        target_h, target_w = size
    _, _, h, w = tensor.shape
    if h == target_h and w == target_w:
        return tensor
    
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    
    return tensor[..., top:top + target_h, left:left + target_w]


def _crop_region(tensor, region):
    """テンソルの指定領域をクロップ"""
    if region is None:
        return tensor
    
    y_start, y_end, x_start, x_end = region
    
    if y_start < 0 or x_start < 0:
        raise ValueError("Crop region start indices must be non-negative.")
    if y_end > tensor.shape[-2] or x_end > tensor.shape[-1]:
        raise ValueError("Crop region exceeds tensor spatial dimensions.")
    if y_end <= y_start or x_end <= x_start:
        raise ValueError("Crop region must have positive height and width.")
    
    return tensor[..., y_start:y_end, x_start:x_end]


class TorchHologramSimulator(nn.Module):
    """PyTorchベースのホログラムシミュレータ
    
    Angular Spectrum法を使用して、ホログラムパターンから
    再生像をシミュレートします。バッチ処理に対応。
    
    Note: FFT計算は常にfp32で行う（精度維持のため）
    
    Args:
        image_shape: 入力画像サイズ (int または tuple)
        distance: 伝播距離 [m]
        pitch: ピクセルピッチ [m]
        wave_length: 光の波長 [m]
        pad_to_natural: 自然サンプリングサイズにパディングするか
        shift: FFTシフトを適用するか
        normalize: 出力を正規化するか
        crop_region: クロップ領域 (y_start, y_end, x_start, x_end)
    
    例:
        >>> sim = TorchHologramSimulator(512, distance=0.5)
        >>> reconstructed = sim(hologram_batch)
    """

    def __init__(
        self,
        image_shape,
        distance,
        pitch=DEFAULT_PITCH,
        wave_length=DEFAULT_WAVE_LENGTH,
        pad_to_natural=True,
        shift=True,
        normalize=False,
        crop_region=None,
    ):
        super().__init__()
        
        if isinstance(image_shape, int):
            height = width = image_shape
        else:
            height, width = image_shape
        
        min_dim = max(height, width)
        if pad_to_natural:
            natural_dim = _natural_size(distance, pitch=pitch, wave_length=wave_length)
            target_dim = max(min_dim, natural_dim)
        else:
            target_dim = min_dim
        
        self.target_shape = (target_dim, target_dim)
        self.pad_to_natural = pad_to_natural
        self.shift = shift
        self.normalize = normalize
        self.crop_region = crop_region
        
        if crop_region is not None:
            y_start, y_end, x_start, x_end = crop_region
            crop_height = y_end - y_start
            crop_width = x_end - x_start
            
            if crop_height <= 0 or crop_width <= 0:
                raise ValueError("Invalid crop region dimensions.")
            max_h, max_w = self.target_shape
            if y_start < 0 or x_start < 0 or y_end > max_h or x_end > max_w:
                raise ValueError("Crop region exceeds propagated field size.")
            self.original_shape = (crop_height, crop_width)
        else:
            self.original_shape = (height, width)
        
        # ゾーンプレートをバッファとして登録
        zone_plate_fft = _build_zone_plate_fft(
            self.target_shape,
            distance,
            pitch=pitch,
            wave_length=wave_length,
        )
        self.register_buffer("zone_plate_fft", zone_plate_fft)
        self.shift_offsets = (self.target_shape[0] // 2, self.target_shape[1] // 2)

    @torch.amp.autocast('cuda', enabled=False)  # FFTは常にfp32で計算
    def forward(self, hologram):
        """ホログラムから再生像をシミュレート
        
        Args:
            hologram: 入力ホログラム (B, 1, H, W)
        
        Returns:
            再生像の振幅 (B, 1, H', W')
        """
        if hologram.ndim != 4:
            raise ValueError("Expected hologram tensor with shape (B, C, H, W)")
        if hologram.shape[1] != 1:
            raise ValueError("TorchHologramSimulator assumes single-channel inputs.")
        
        # fp16入力をfp32に変換
        hologram = hologram.float()
        
        padded = _pad_tensor(hologram, self.target_shape)
        hologram_complex = padded.to(torch.complex64).squeeze(1)
        
        zone_plate_fft = self.zone_plate_fft
        # Handle case where DataParallel/buffer persistence converts complex to real with last dim 2
        if zone_plate_fft.ndim == 3 and zone_plate_fft.shape[-1] == 2:
            zone_plate_fft = torch.view_as_complex(zone_plate_fft)
        
        zone_plate_fft = zone_plate_fft.to(hologram_complex.dtype).unsqueeze(0)
        wave_front = torch.fft.ifft2(torch.fft.fft2(hologram_complex) * zone_plate_fft)
        
        if self.shift:
            wave_front = torch.roll(wave_front, shifts=self.shift_offsets, dims=(-2, -1))
        
        amplitude = wave_front.abs().unsqueeze(1)
        
        if self.normalize:
            amplitude = amplitude / (amplitude.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        
        if self.crop_region is not None:
            amplitude = _crop_region(amplitude, self.crop_region)
        else:
            amplitude = _center_crop(amplitude, self.original_shape)
        
        return amplitude


def simulate_hologram_batch(
    holograms,
    simulator,
    normalize=True,
    mean_center=True,
):
    """バッチ単位でホログラムシミュレーションを実行
    
    Args:
        holograms: ホログラムバッチ (B, 1, H, W)
        simulator: TorchHologramSimulatorインスタンス
        normalize: 出力を正規化するか
        mean_center: 入力の平均を0にするか
    
    Returns:
        再生像の振幅バッチ (B, 1, H', W')
    """
    if holograms.ndim != 4:
        raise ValueError("Expected holograms with shape (B, C, H, W)")
    
    if mean_center:
        holograms = holograms - holograms.mean(dim=(-2, -1), keepdim=True)
    
    amplitudes = simulator(holograms)
    
    # Handle DataParallel wrapping
    sim_module = getattr(simulator, "module", simulator)
    
    if normalize and not sim_module.normalize:
        amplitudes = amplitudes / (amplitudes.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    
    return amplitudes
