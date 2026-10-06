import argparse
import json
import time
from pathlib import Path

import rasterio

from satcoreg import catalog
from satcoreg.grid import read_target, to_gray, valid_mask
from satcoreg.match import MatchParams, find_tie_points, read_geojson, summarize, write_geojson
from satcoreg.reference import build_reference
from satcoreg.warp import warp_image


def _out_dir(base: Path, image: Path) -> Path:
    return base / image.stem


def cmd_list(args: argparse.Namespace) -> None:
    res = catalog.filter_resources(catalog.list_resources(args.dataset), args.date, args.mesh)
    for r in res:
        print(f"{r.name}\t{r.size / 1e6:8.1f} MB")
    print(f"{len(res)} 件 / 合計 {sum(r.size for r in res) / 1e9:.1f} GB")


def cmd_download(args: argparse.Namespace) -> None:
    res = catalog.filter_resources(catalog.list_resources(args.dataset), args.date, args.mesh)
    for i, r in enumerate(res, 1):
        print(f"[{i}/{len(res)}] {r.name} ({r.size / 1e6:.0f} MB)")
        catalog.download(r, args.out)


def match_one(image: Path, out_dir: Path, args: argparse.Namespace, ref_path: Path) -> dict:
    t0 = time.perf_counter()
    grid, tgt, tgt_valid = read_target(str(image), args.work_res)
    if ref_path.exists():
        with rasterio.open(ref_path) as src:
            ref_rgb = src.read()
    else:
        ref_rgb = build_reference(grid, ref_path, zoom=args.zoom, cache_dir=args.cache)
    params = MatchParams(spacing=args.spacing, window=args.window)
    points = find_tie_points(grid, tgt, tgt_valid, to_gray(ref_rgb), valid_mask(ref_rgb), params)
    summary = summarize(points) | {"seconds": round(time.perf_counter() - t0, 1)}
    return {"grid": grid, "points": points, "summary": summary}


def cmd_match(args: argparse.Namespace) -> None:
    for image in args.images:
        out = _out_dir(args.out, image)
        print(f"== {image.name}: マッチング")
        m = match_one(image, out, args, out / "reference.tif")
        write_geojson(m["points"], m["grid"].crs, out / "tiepoints.geojson")
        (out / "match_summary.json").write_text(
            json.dumps(m["summary"], ensure_ascii=False, indent=2)
        )
        print(json.dumps(m["summary"], ensure_ascii=False))


def cmd_warp(args: argparse.Namespace) -> None:
    for image in args.images:
        out = _out_dir(args.out, image)
        gcps = args.tiepoints or out / "tiepoints.geojson"
        with rasterio.open(image) as src:
            pts = read_geojson(gcps, src.crs)
        if len(pts) < args.min_points:
            print(f"== {image.name}: 有効なタイポイントが {len(pts)} 点しかないため補正しません")
            continue
        dst = out / f"{image.stem}_coreg.tif"
        print(f"== {image.name}: {len(pts)} 点で補正 → {dst}")
        t0 = time.perf_counter()
        warp_image(image, pts, dst, smoothing=args.smoothing, compress=args.compress)
        print(f"  {time.perf_counter() - t0:.0f} 秒")
        if args.verify:
            print("  補正後の残差を確認")
            m = match_one(dst, out, args, out / "reference.tif")
            write_geojson(m["points"], m["grid"].crs, out / "verify.geojson")
            (out / "verify_summary.json").write_text(
                json.dumps(m["summary"], ensure_ascii=False, indent=2)
            )
            print("  " + json.dumps(m["summary"], ensure_ascii=False))


def cmd_run(args: argparse.Namespace) -> None:
    cmd_match(args)
    cmd_warp(args)


def main() -> None:
    p = argparse.ArgumentParser(prog="satcoreg", description="衛星画像を地理院写真に合わせ込む")
    sub = p.add_subparsers(required=True)

    def add_catalog_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", default=catalog.DEFAULT_DATASET)
        sp.add_argument("--date", nargs="*", help="撮像日 (例: 20261001)")
        sp.add_argument("--mesh", nargs="*", help="図郭番号 (例: 5339K4)")

    sp = sub.add_parser("list", help="データセットの GeoTIFF 一覧")
    add_catalog_args(sp)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("download", help="GeoTIFF をダウンロード")
    add_catalog_args(sp)
    sp.add_argument("--out", type=Path, default=Path("data/raw"))
    sp.set_defaults(func=cmd_download)

    def add_match_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--work-res", type=float, default=1.0, help="マッチングの解像度 (m)")
        sp.add_argument("--zoom", type=int, default=17, help="参照に使う地理院タイルのズーム")
        sp.add_argument("--spacing", type=int, default=150, help="格子間隔（作業画素）")
        sp.add_argument("--window", type=int, default=256, help="窓サイズ（作業画素）")
        sp.add_argument("--cache", type=Path, default=Path("data/cache/tiles"))

    def add_warp_args(sp: argparse.ArgumentParser, tiepoints: bool) -> None:
        if tiepoints:
            sp.add_argument(
                "--tiepoints", type=Path, help="既定は <out>/<画像名>/tiepoints.geojson"
            )
        sp.add_argument("--smoothing", type=float, default=0.1, help="TPS の平滑化")
        sp.add_argument("--min-points", type=int, default=10)
        sp.add_argument("--compress", default="DEFLATE", help="COG の圧縮 (DEFLATE / JPEG など)")
        sp.add_argument("--no-verify", dest="verify", action="store_false")

    for name, func, helptext in [
        ("match", cmd_match, "タイポイントを抽出"),
        ("warp", cmd_warp, "タイポイントで補正して COG を出力"),
        ("run", cmd_run, "match と warp を続けて実行"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("images", nargs="+", type=Path)
        sp.add_argument("--out", type=Path, default=Path("data/out"))
        add_match_args(sp)
        if name != "match":
            add_warp_args(sp, tiepoints=(name == "warp"))
        if name == "run":
            sp.set_defaults(tiepoints=None)
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)
