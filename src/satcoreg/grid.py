"""対象画像を縮小した「作業グリッド」の定義と読み込み。

マッチングは全解像度（0.4 m）ではなく、参照写真に近い解像度（既定 1 m）で行う。
作業グリッドは対象画像のアフィン変換を等倍率で縮小したもので、
対象画像と同じ範囲・同じ向きを持つ。
"""

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling


@dataclass(frozen=True)
class WorkGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def res(self) -> float:
        return abs(self.transform.a)


def work_grid_for(src: rasterio.DatasetReader, work_res: float) -> WorkGrid:
    factor = work_res / abs(src.transform.a)
    width = int(np.ceil(src.width / factor))
    height = int(np.ceil(src.height / factor))
    transform = src.transform @ Affine.scale(src.width / width, src.height / height)
    return WorkGrid(src.crs, transform, width, height)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) uint8 → (H, W) float32 の輝度。"""
    r, g, b = (rgb[i].astype(np.float32) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def valid_mask(rgb: np.ndarray) -> np.ndarray:
    """欠損（真っ黒・真っ白）でない画素。データセットに nodata 指定がないため値で判定する。"""
    black = np.all(rgb <= 1, axis=0)
    white = np.all(rgb >= 254, axis=0)
    return ~(black | white)


def read_target(path: str, work_res: float) -> tuple[WorkGrid, np.ndarray, np.ndarray]:
    """対象画像を作業グリッドに縮小して読む。戻り値は (grid, gray, valid)。"""
    with rasterio.open(path) as src:
        grid = work_grid_for(src, work_res)
        rgb = src.read(
            [1, 2, 3], out_shape=(3, grid.height, grid.width), resampling=Resampling.average
        )
    return grid, to_gray(rgb), valid_mask(rgb)
