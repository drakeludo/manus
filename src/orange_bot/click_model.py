from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ClickPoint:
    x: int
    y: int
    radius: int = 16


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class OrangeClickNet(nn.Module):
    def __init__(self, base_channels: int = 24):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6

        self.stem = ConvBlock(3, c1)
        self.down1 = ConvBlock(c1, c2, stride=2)
        self.down2 = ConvBlock(c2, c3, stride=2)
        self.down3 = ConvBlock(c3, c4, stride=2)
        self.bottleneck = ConvBlock(c4, c4)

        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)

        self.center_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.mask_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        bottleneck = self.bottleneck(down3)

        up2 = self.up2(bottleneck, down2)
        up1 = self.up1(up2, down1)
        up0 = self.up0(up1, stem)

        return {
            "center_heatmap": self.center_head(up0),
            "orange_mask": self.mask_head(up0),
        }


def build_center_heatmap(height: int, width: int, points: Iterable[ClickPoint], sigma: float = 6.0) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    if height <= 0 or width <= 0:
        return heatmap

    yy, xx = np.mgrid[0:height, 0:width]
    sigma_sq = max(1e-6, sigma * sigma)
    for point in points:
        gaussian = np.exp(-((xx - point.x) ** 2 + (yy - point.y) ** 2) / (2.0 * sigma_sq))
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return np.clip(heatmap, 0.0, 1.0)


def build_click_mask(height: int, width: int, points: Iterable[ClickPoint]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for point in points:
        cv2.circle(mask, (int(point.x), int(point.y)), int(max(4, point.radius)), 1.0, thickness=-1)
    return np.clip(mask, 0.0, 1.0)


def extract_click_points(
    center_heatmap: np.ndarray,
    orange_mask: np.ndarray,
    confidence_threshold: float = 0.18,
    mask_threshold: float = 0.18,
    min_distance: int = 18,
    max_points: int = 96,
) -> list[tuple[int, int, float]]:
    if center_heatmap.size == 0:
        return []

    pooled = cv2.dilate(center_heatmap, np.ones((5, 5), dtype=np.uint8), iterations=1)
    peak_mask = (center_heatmap >= confidence_threshold) & (center_heatmap >= pooled - 1e-6)
    ys, xs = np.where(peak_mask)

    candidates: list[tuple[float, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        mask_score = float(orange_mask[y, x]) if orange_mask.size else 0.0
        if mask_score < mask_threshold:
            continue
        score = float(center_heatmap[y, x] + 0.35 * mask_score)
        candidates.append((score, x, y))

    candidates.sort(reverse=True)
    points: list[tuple[int, int, float]] = []
    for score, x, y in candidates:
        if any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 < min_distance for px, py, _ in points):
            continue
        points.append((x, y, score))
        if len(points) >= max_points:
            break
    return points
