"""タイポイントのずれベクトルを参照写真の上に描く。

緑=採用、橙=近傍判定で除外、赤=全体判定で除外。矢印の長さはずれの 10 倍。

uv run python scripts/plot_tiepoints.py data/out/<画像名> tiepoints.geojson out.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

ARROW_SCALE = 10.0
COLORS = {"ok": (0, 255, 0), "local_outlier": (0, 165, 255), "global_outlier": (0, 0, 255)}


def main() -> None:
    out_dir, name, dst = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    with rasterio.open(out_dir / "reference.tif") as src:
        ref = src.read(out_shape=(3, src.height // 8, src.width // 8))
        f = src.width / ref.shape[2]
    img = np.ascontiguousarray(ref.transpose(1, 2, 0)[..., ::-1]) // 2
    fc = json.loads((out_dir / name).read_text("utf-8"))
    for feat in fc["features"]:
        p = feat["properties"]
        if p["status"] not in COLORS:
            continue
        x, y = p["col"] / f, p["row"] / f
        x2, y2 = x + p["de"] * ARROW_SCALE / f, y - p["dn"] * ARROW_SCALE / f
        cv2.arrowedLine(
            img, (int(x), int(y)), (int(x2), int(y2)), COLORS[p["status"]], 1, tipLength=0.2
        )
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
