"""G空間情報センター（CKAN）のデータセットから GeoTIFF の一覧取得とダウンロードを行う。"""

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

CKAN_API = "https://www.geospatial.jp/ckan/api/3/action/package_show"
DEFAULT_DATASET = "25"
NAME_RE = re.compile(r"^(?P<mesh>\w{6})_(?P<date>\d{8})_(?P<zone>\d{2}N)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class Resource:
    name: str
    mesh: str
    date: str
    url: str
    size: int


def list_resources(dataset: str = DEFAULT_DATASET) -> list[Resource]:
    resp = httpx.get(CKAN_API, params={"id": dataset}, timeout=60)
    resp.raise_for_status()
    out = []
    for r in resp.json()["result"]["resources"]:
        m = NAME_RE.match(r["name"])
        if m:
            out.append(
                Resource(r["name"], m["mesh"].upper(), m["date"], r["url"], int(r.get("size") or 0))
            )
    return out


def filter_resources(
    resources: list[Resource], dates: list[str] | None, meshes: list[str] | None
) -> list[Resource]:
    meshes_u = [m.upper() for m in meshes] if meshes else None
    return [
        r
        for r in resources
        if (not dates or r.date in dates) and (not meshes_u or r.mesh in meshes_u)
    ]


def download(res: Resource, out_dir: Path) -> Path:
    """途中で切れても続きから取得する。サイズが一致していれば取得しない。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / res.name
    have = path.stat().st_size if path.exists() else 0
    if res.size and have == res.size:
        return path
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", res.url, headers=headers, timeout=120, follow_redirects=True) as resp:
        if resp.status_code == 200:
            have = 0  # Range 非対応なら最初から
        elif resp.status_code != 206:
            resp.raise_for_status()
        with path.open("ab" if have else "wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
    if res.size and path.stat().st_size != res.size:
        raise OSError(f"{res.name}: サイズ不一致 ({path.stat().st_size} != {res.size})")
    return path
