"""地理院タイル（シームレス空中写真）から、作業グリッド上の参照画像を作る。"""

import asyncio
import math
from pathlib import Path

import cv2
import httpx
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from satcoreg.grid import WorkGrid

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
TILE_SIZE = 256
WEB_MERCATOR_HALF = 20037508.342789244
USER_AGENT = "satcoreg/0.1 (satellite image co-registration)"


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


async def _fetch_tiles(
    tiles: list[tuple[int, int]], z: int, cache_dir: Path, concurrency: int
) -> dict[tuple[int, int], np.ndarray | None]:
    sem = asyncio.Semaphore(concurrency)
    results: dict[tuple[int, int], np.ndarray | None] = {}

    async def fetch(client: httpx.AsyncClient, x: int, y: int) -> None:
        path = cache_dir / str(z) / str(x) / f"{y}.jpg"
        missing = path.with_suffix(".404")
        if missing.exists():
            results[(x, y)] = None
            return
        if not path.exists():
            async with sem:
                for attempt in range(4):
                    try:
                        resp = await client.get(TILE_URL.format(z=z, x=x, y=y))
                        break
                    except httpx.TransportError:
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2**attempt)
            path.parent.mkdir(parents=True, exist_ok=True)
            if resp.status_code == 404:
                missing.touch()
                results[(x, y)] = None
                return
            resp.raise_for_status()
            path.write_bytes(resp.content)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        results[(x, y)] = None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        await asyncio.gather(*(fetch(client, x, y) for x, y in tiles))
    return results


def build_reference(
    grid: WorkGrid,
    out_path: Path,
    zoom: int = 17,
    cache_dir: Path = Path("data/cache/tiles"),
    concurrency: int = 8,
) -> np.ndarray:
    """作業グリッドを覆う参照画像 (3, H, W) uint8 を作り、GeoTIFF にも保存する。"""
    left, top = grid.transform @ (0, 0)
    right, bottom = grid.transform @ (grid.width, grid.height)
    w, s, e, n = transform_bounds(
        grid.crs,
        "EPSG:4326",
        min(left, right),
        min(top, bottom),
        max(left, right),
        max(top, bottom),
    )
    x0, y0 = _lonlat_to_tile(w, n, zoom)
    x1, y1 = _lonlat_to_tile(e, s, zoom)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    print(f"  参照タイル z{zoom}: {len(tiles)} 枚")
    images = asyncio.run(_fetch_tiles(tiles, zoom, cache_dir, concurrency))

    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.zeros((3, ny * TILE_SIZE, nx * TILE_SIZE), dtype=np.uint8)
    for (x, y), img in images.items():
        if img is not None:
            r, c = (y - y0) * TILE_SIZE, (x - x0) * TILE_SIZE
            mosaic[:, r : r + TILE_SIZE, c : c + TILE_SIZE] = img.transpose(2, 0, 1)

    tile_m = 2 * WEB_MERCATOR_HALF / 2**zoom
    merc_transform = Affine(
        tile_m / TILE_SIZE, 0, -WEB_MERCATOR_HALF + x0 * tile_m,
        0, -tile_m / TILE_SIZE, WEB_MERCATOR_HALF - y0 * tile_m,
    )  # fmt: skip
    ref = np.zeros((3, grid.height, grid.width), dtype=np.uint8)
    reproject(
        mosaic,
        ref,
        src_transform=merc_transform,
        src_crs="EPSG:3857",
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.bilinear,
        src_nodata=0,
        dst_nodata=0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", width=grid.width, height=grid.height, count=3,
        dtype="uint8", crs=grid.crs, transform=grid.transform, nodata=0,
        tiled=True, compress="deflate",
    ) as dst:  # fmt: skip
        dst.write(ref)
    return ref
