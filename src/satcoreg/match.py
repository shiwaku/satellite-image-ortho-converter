"""格子状の窓ごとの位相相関でタイポイントを求め、外れ値を除く。

考え方は AROSICS の COREG_LOCAL と同じ。
1. 作業グリッド上に等間隔の格子点を置く
2. 各点で対象画像と参照画像から窓を切り出し、位相相関で相対ずれを求める
3. 応答値・整合後の相関係数・近傍との一貫性・全体アフィンからの乖離で外れ値を除く

ずれ (dE, dN) は「対象画像上の地物位置 + (dE, dN) = 参照（地図）上の位置」となる量（m）。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from rasterio.warp import transform as transform_coords
from scipy.spatial import cKDTree

from satcoreg.grid import WorkGrid


@dataclass
class TiePoint:
    col: float  # 作業グリッド上の窓中心（対象画像側）
    row: float
    e: float  # 対象画像上の座標 (m)
    n: float
    de: float  # 補正量 (m)
    dn: float
    response: float
    ncc: float
    status: str = "ok"


@dataclass
class MatchParams:
    spacing: int = 150  # 格子間隔（作業グリッドの画素）
    window: int = 256  # 窓サイズ（作業グリッドの画素）
    min_valid: float = 0.9  # 窓内の有効画素率の下限
    min_std: float = 6.0  # 窓内の輝度標準偏差の下限（水面など無地を除く）
    min_response: float = 0.05
    min_ncc: float = 0.3
    max_shift: float = 0.25  # 窓サイズに対する比率の上限
    local_tol_px: float = 2.0  # 近傍中央値からの乖離の許容下限（画素）
    global_tol_px: float = 15.0  # 全体の中央値からの乖離の許容下限（画素）
    neighbors: int = 8  # 近傍判定に使う点数


def _prep(win: np.ndarray) -> np.ndarray:
    win = cv2.GaussianBlur(win, (0, 0), 1.0)
    return ((win - win.mean()) / (win.std() + 1e-6)).astype(np.float32)


ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)


def estimate_shift(ref_win: np.ndarray, tgt_win: np.ndarray) -> tuple[float, float, float, float]:
    """tgt(x) ≈ ref(x - s) となる s=(sx, sy) と、位相相関の応答値・整合後の相関係数を返す。

    位相相関で粗く求めたあと、ECC（平行移動のみ）でサブピクセルまで詰める。
    OpenCV の phaseCorrelate のサブピクセル推定は偏りが大きいため。
    ECC が収束しなければ相関係数を 0 として返し、後段で落とす。
    """
    ref_p, tgt_p = _prep(ref_win), _prep(tgt_win)
    hann = cv2.createHanningWindow(ref_p.shape[::-1], cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(ref_p, tgt_p, hann)
    # 端の影響を避けるため、ECC は中央部の参照に対して行う
    k = ref_p.shape[0] // 8
    template = np.ascontiguousarray(ref_p[k:-k, k:-k])
    warp = np.float32([[1, 0, sx + k], [0, 1, sy + k]])
    try:
        cc, warp = cv2.findTransformECC(
            template, tgt_p, warp, cv2.MOTION_TRANSLATION, ECC_CRITERIA, None, 1
        )
    except cv2.error:
        return sx, sy, response, 0.0
    return float(warp[0, 2] - k), float(warp[1, 2] - k), response, float(cc)


def find_tie_points(
    grid: WorkGrid,
    tgt: np.ndarray,
    tgt_valid: np.ndarray,
    ref: np.ndarray,
    ref_valid: np.ndarray,
    p: MatchParams,
) -> list[TiePoint]:
    half = p.window // 2
    points: list[TiePoint] = []
    for r in range(half, grid.height - half, p.spacing):
        for c in range(half, grid.width - half, p.spacing):
            sl = (slice(r - half, r + half), slice(c - half, c + half))
            if tgt_valid[sl].mean() < p.min_valid or ref_valid[sl].mean() < p.min_valid:
                continue
            tw, rw = tgt[sl], ref[sl]
            if tw.std() < p.min_std or rw.std() < p.min_std:
                continue
            sx, sy, resp, ncc = estimate_shift(rw, tw)
            e, n = grid.transform @ (c, r)
            pt = TiePoint(c, r, e, n, -sx * grid.res, sy * grid.res, resp, ncc)
            if resp < p.min_response:
                pt.status = "low_response"
            elif ncc < p.min_ncc:
                pt.status = "low_ncc"
            elif max(abs(sx), abs(sy)) > p.max_shift * p.window:
                pt.status = "too_large"
            points.append(pt)
    _reject_global(points, grid.res, p)
    _reject_local(points, grid.res, p)
    return points


def _reject_global(points: list[TiePoint], res: float, p: MatchParams) -> None:
    """全体の中央値から大きく外れた点を落とす。

    アフィンの RANSAC は点が偏在する図郭（海が広いなど）で外挿が傾き、
    正しい点まで落とすため使わない。ここでは明らかな誤対応だけを除き、細かい判定は近傍で行う。
    """
    ok = [pt for pt in points if pt.status == "ok"]
    if len(ok) < 6:
        return
    d = np.array([[pt.de, pt.dn] for pt in ok])
    dist = np.linalg.norm(d - np.median(d, axis=0), axis=1)
    tol = max(p.global_tol_px * res, 4 * 1.4826 * np.median(dist))
    for pt, x in zip(ok, dist, strict=True):
        if x > tol:
            pt.status = "global_outlier"


def _reject_local(points: list[TiePoint], res: float, p: MatchParams, iterations: int = 3) -> None:
    """近傍 k 点の中央値と比べて外れた点を落とす。点がまばらな所でも k 点は必ず比べる。"""
    for _ in range(iterations):
        ok = [pt for pt in points if pt.status == "ok"]
        if len(ok) <= p.neighbors:
            return
        xy = np.array([[pt.e, pt.n] for pt in ok])
        d = np.array([[pt.de, pt.dn] for pt in ok])
        _, idx = cKDTree(xy).query(xy, k=p.neighbors + 1)
        rejected = []
        for i, nbrs in enumerate(idx[:, 1:]):
            med = np.median(d[nbrs], axis=0)
            dev = np.linalg.norm(d[nbrs] - med, axis=1)
            tol = max(p.local_tol_px * res, 3 * 1.4826 * np.median(dev))
            if np.linalg.norm(d[i] - med) > tol:
                rejected.append(ok[i])
        if not rejected:
            return
        for pt in rejected:
            pt.status = "local_outlier"


def summarize(points: list[TiePoint]) -> dict:
    ok = [pt for pt in points if pt.status == "ok"]
    counts: dict[str, int] = {}
    for pt in points:
        counts[pt.status] = counts.get(pt.status, 0) + 1
    out: dict = {"candidates": len(points), "status": counts}
    if ok:
        mag = np.hypot([pt.de for pt in ok], [pt.dn for pt in ok])
        out |= {
            "shift_m_median": round(float(np.median(mag)), 2),
            "shift_m_p90": round(float(np.percentile(mag, 90)), 2),
            "shift_m_max": round(float(mag.max()), 2),
            "de_m_median": round(float(np.median([pt.de for pt in ok])), 2),
            "dn_m_median": round(float(np.median([pt.dn for pt in ok])), 2),
        }
    return out


def write_geojson(points: list[TiePoint], crs, path: Path) -> None:
    """QGIS で確認・編集できるよう、経緯度の GeoJSON で書く。

    use 属性を 0/1 で切り替えれば、warp で使う点を手で選べる。
    手で点を足す場合は、地図上の正しい位置ではなく「対象画像上の地物位置」に点を置き、
    de/dn（m、UTM の東・北方向）に補正量を入れる。
    """
    lon, lat = transform_coords(crs, "EPSG:4326", [pt.e for pt in points], [pt.n for pt in points])
    features = []
    for pt, x, y in zip(points, lon, lat, strict=True):
        props = asdict(pt) | {"use": int(pt.status == "ok")}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(x, 8), round(y, 8)]},
                "properties": props,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), "utf-8")


def read_geojson(path: Path, crs) -> np.ndarray:
    """use=1 の点を (E, N, dE, dN) の配列で返す。座標は QGIS で動かした場合も考慮してジオメトリから取る。"""
    fc = json.loads(path.read_text("utf-8"))
    rows = [f for f in fc["features"] if int(f["properties"].get("use", 0)) == 1]
    lon = [f["geometry"]["coordinates"][0] for f in rows]
    lat = [f["geometry"]["coordinates"][1] for f in rows]
    e, n = transform_coords("EPSG:4326", crs, lon, lat)
    de = [float(f["properties"]["de"]) for f in rows]
    dn = [float(f["properties"]["dn"]) for f in rows]
    return np.column_stack([e, n, de, dn])
