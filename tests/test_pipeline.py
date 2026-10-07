import numpy as np
import rasterio
from affine import Affine
from scipy.ndimage import gaussian_filter, shift

from satcoreg.grid import WorkGrid, read_target, to_gray
from satcoreg.match import MatchParams, coarse_prior, estimate_shift, find_tie_points
from satcoreg.warp import warp_image

CRS = "EPSG:32654"


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=(h, w)), 1)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 200 + 20).astype(np.float32)


def test_estimate_shift_sign():
    ref = _texture(256, 256)
    # 地物が西に 1.3 px、南に 2.5 px ずれて写った対象画像
    tgt = shift(ref, (2.5, -1.3), order=3, mode="nearest")
    sx, sy, _, ncc = estimate_shift(ref, tgt)
    assert abs(sx + 1.3) < 0.1 and abs(sy - 2.5) < 0.1
    assert ncc > 0.9


def test_tie_points_recover_offset():
    h = w = 1200
    ref = _texture(h, w, seed=1)
    tgt = shift(ref, (-4, 5), order=1, mode="nearest")  # 北に 4 px、東に 5 px
    grid = WorkGrid(rasterio.crs.CRS.from_string(CRS), Affine(1, 0, 400000, 0, -1, 3940000), w, h)
    valid = np.ones((h, w), bool)
    pts = find_tie_points(grid, tgt, valid, ref, valid, MatchParams(spacing=150, window=256))
    ok = [p for p in pts if p.status == "ok"]
    assert len(ok) > 20
    # 地図に合わせるには西へ 5 m、南へ 4 m 戻す
    assert abs(np.median([p.de for p in ok]) + 5) < 0.2
    assert abs(np.median([p.dn for p in ok]) + 4) < 0.2


def test_warp_moves_image_back(tmp_path):
    h = w = 800
    base = _texture(h, w, seed=2)
    tgt = shift(base, (-4, 5), order=1, mode="nearest")
    transform = Affine(0.4, 0, 400000, 0, -0.4, 3940000)
    src_path = tmp_path / "in.tif"
    with rasterio.open(
        src_path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
        crs=CRS, transform=transform,
    ) as dst:  # fmt: skip
        dst.write(np.repeat(tgt[None].astype(np.uint8), 3, axis=0))

    # 一様に西へ 2.0 m（5 px）、南へ 1.6 m（4 px）戻す補正点
    e, n = np.meshgrid(np.linspace(400010, 400310, 5), np.linspace(3939690, 3939990, 5))
    pts = np.column_stack([e.ravel(), n.ravel(), np.full(25, -2.0), np.full(25, -1.6)])
    out_path = tmp_path / "out.tif"
    warp_image(src_path, pts, out_path, smoothing=0.0)

    _, out_gray, _, _ = read_target(str(out_path), 0.4, mask_clouds=False)
    inner = (slice(50, -50), slice(50, -50))
    diff = np.abs(
        out_gray[inner] - to_gray(np.repeat(base[None].astype(np.uint8), 3, axis=0))[inner]
    )
    assert diff.mean() < 2.0


def test_coarse_prior_recovers_large_offset():
    h = w = 2400
    rng = np.random.default_rng(3)
    # 縮小しても模様が残るよう、細かい模様と粗い模様を重ねる
    img = gaussian_filter(rng.normal(size=(h, w)), 1) + 3 * gaussian_filter(
        rng.normal(size=(h, w)), 8
    )
    ref = ((img - img.min()) / (img.max() - img.min()) * 200 + 20).astype(np.float32)
    tgt = shift(ref, (-45, 70), order=1, mode="nearest")  # 北に 45 px、東に 70 px
    grid = WorkGrid(rasterio.crs.CRS.from_string(CRS), Affine(1, 0, 400000, 0, -1, 3940000), w, h)
    valid = np.ones((h, w), bool)
    params = MatchParams(spacing=200, window=256)

    without = find_tie_points(grid, tgt, valid, ref, valid, params)
    assert sum(p.status == "ok" for p in without) < 5  # 窓の 1/4 を超えるずれは拾えない

    prior, _ = coarse_prior(grid, tgt, valid, ref, valid)
    assert prior is not None
    ok = [
        p
        for p in find_tie_points(grid, tgt, valid, ref, valid, params, prior=prior)
        if p.status == "ok"
    ]
    assert len(ok) > 50
    assert abs(np.median([p.de for p in ok]) + 70) < 0.2
    assert abs(np.median([p.dn for p in ok]) + 45) < 0.2
