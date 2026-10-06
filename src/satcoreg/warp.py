"""タイポイントの補正量から変位場を作り、元の解像度・グリッドのまま画像を引き直す。

変位場は薄板スプライン（TPS）で補間する。GDAL の gdalwarp -tps は GCP を厳密に通すが、
ここでは smoothing を与えて、マッチング誤差（数十 cm 程度）で局所的に歪むのを抑える。
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.shutil
from rasterio.windows import Window
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

COARSE_STEP = 64  # 変位場を評価する間隔（出力画素）
STRIP_ROWS = 1024


def displacement_field(
    pts: np.ndarray, transform, height: int, width: int, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    """出力画素の粗いグリッド上で補正量 (dE, dN) を評価する。pts は (E, N, dE, dN)。"""
    origin = pts[:, :2].mean(axis=0)
    xy_km = (pts[:, :2] - origin) / 1000.0
    rbf = RBFInterpolator(xy_km, pts[:, 2:], kernel="thin_plate_spline", smoothing=smoothing)
    rows = np.arange(0, height + COARSE_STEP, COARSE_STEP, dtype=np.float64)
    cols = np.arange(0, width + COARSE_STEP, COARSE_STEP, dtype=np.float64)
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
    e, n = transform @ (cc.ravel(), rr.ravel())
    q = (np.column_stack([e, n]) - origin) / 1000.0
    d = rbf(q).reshape(len(rows), len(cols), 2)
    return d[..., 0], d[..., 1]


def warp_image(
    src_path: Path,
    pts: np.ndarray,
    out_path: Path,
    smoothing: float = 0.1,
    compress: str = "DEFLATE",
) -> None:
    with rasterio.open(src_path) as src:
        de_grid, dn_grid = displacement_field(pts, src.transform, src.height, src.width, smoothing)
        res_x, res_y = abs(src.transform.a), abs(src.transform.e)
        profile = src.profile | {
            "driver": "GTiff", "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "DEFLATE", "predictor": 2, "BIGTIFF": "IF_SAFER",
        }  # fmt: skip
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
            tmp_path = Path(tmp) / "warped.tif"
            with rasterio.open(tmp_path, "w", **profile) as dst:
                cols = np.arange(src.width, dtype=np.float32)
                for r0 in range(0, src.height, STRIP_ROWS):
                    r1 = min(r0 + STRIP_ROWS, src.height)
                    rr, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float32), cols, indexing="ij")
                    gi, gj = rr / COARSE_STEP, cc / COARSE_STEP
                    de = map_coordinates(de_grid, [gi, gj], order=1, mode="nearest")
                    dn = map_coordinates(dn_grid, [gi, gj], order=1, mode="nearest")
                    # 出力画素の地図座標 X に、元画像上で X - d にあった画素を持ってくる
                    map_c = (cc - de / res_x).astype(np.float32)
                    map_r = (rr + dn / res_y).astype(np.float32)
                    s0 = max(int(np.floor(map_r.min())) - 2, 0)
                    s1 = min(int(np.ceil(map_r.max())) + 3, src.height)
                    if s1 <= s0:
                        continue
                    block = src.read(window=Window(0, s0, src.width, s1 - s0))
                    hwc = np.ascontiguousarray(block.transpose(1, 2, 0))
                    out = cv2.remap(
                        hwc, map_c, map_r - s0, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )  # fmt: skip
                    out = out.reshape(r1 - r0, src.width, -1).transpose(2, 0, 1)
                    dst.write(out, window=Window(0, r0, src.width, r1 - r0))
            rasterio.shutil.copy(
                tmp_path, out_path, driver="COG", compress=compress, blocksize=512,
                overview_resampling="average", bigtiff="IF_SAFER",
                **({"predictor": 2} if compress.upper() == "DEFLATE" else {}),
            )  # fmt: skip
