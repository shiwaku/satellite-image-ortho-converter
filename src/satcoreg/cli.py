import argparse
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import rasterio

from satcoreg import catalog, report
from satcoreg.grid import read_target, to_gray, valid_mask
from satcoreg.match import (
    MatchParams,
    coarse_prior,
    find_tie_points,
    read_geojson,
    summarize,
    write_geojson,
)
from satcoreg.reference import build_reference
from satcoreg.warp import warp_image


def _out_dir(base: Path, image: Path) -> Path:
    return base / image.stem


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


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


def match_one(image: Path, args: argparse.Namespace, ref_path: Path) -> dict:
    t0 = time.perf_counter()
    grid, tgt, tgt_valid, cloud_frac = read_target(str(image), args.work_res)
    if ref_path.exists():
        with rasterio.open(ref_path) as src:
            ref_rgb = src.read()
    else:
        ref_rgb = build_reference(
            grid, ref_path, zoom=args.zoom, cache_dir=args.cache, concurrency=args.tile_concurrency
        )
    ref, ref_valid = to_gray(ref_rgb), valid_mask(ref_rgb)
    prior, coarse = coarse_prior(grid, tgt, tgt_valid, ref, ref_valid)
    params = MatchParams(spacing=args.spacing, window=args.window)
    points = find_tie_points(grid, tgt, tgt_valid, ref, ref_valid, params, prior=prior)
    summary = summarize(points) | {
        "cloud_frac": round(cloud_frac, 3),
        "coarse_ok": sum(pt.status == "ok" for pt in coarse),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return {"grid": grid, "points": points, "summary": summary}


def match_image(image: Path, args: argparse.Namespace) -> None:
    out = _out_dir(args.out, image)
    print(f"== {image.name}: マッチング")
    m = match_one(image, args, out / "reference.tif")
    write_geojson(m["points"], m["grid"].crs, out / "tiepoints.geojson")
    _write_json(out / "match_summary.json", m["summary"])
    print("  " + json.dumps(m["summary"], ensure_ascii=False))


def warp_one(image: Path, args: argparse.Namespace) -> None:
    out = _out_dir(args.out, image)
    gcps = args.tiepoints or out / "tiepoints.geojson"
    with rasterio.open(image) as src:
        pts = read_geojson(gcps, src.crs)
    if len(pts) < args.min_points:
        msg = f"有効なタイポイントが {len(pts)} 点しかないため補正しません"
        print(f"== {image.name}: {msg}")
        _write_json(out / "status.json", {"status": "skipped", "reason": msg})
        return
    dst = out / f"{image.stem}_coreg.tif"
    print(f"== {image.name}: {len(pts)} 点で補正 → {dst}")
    t0 = time.perf_counter()
    warp_image(image, pts, dst, smoothing=args.smoothing, compress=args.compress)
    seconds = round(time.perf_counter() - t0, 1)
    print(f"  {seconds:.0f} 秒")
    if args.verify:
        print("  補正後の残差を確認")
        m = match_one(dst, args, out / "reference.tif")
        write_geojson(m["points"], m["grid"].crs, out / "verify.geojson")
        _write_json(out / "verify_summary.json", m["summary"])
        print("  " + json.dumps(m["summary"], ensure_ascii=False))
    _write_json(out / "status.json", {"status": "done", "warp_seconds": seconds})


def run_one(image: Path, args: argparse.Namespace) -> None:
    """1 図郭を match → warp。処理済みは飛ばし、失敗しても例外を外へ出さない。"""
    status = _out_dir(args.out, image) / "status.json"
    if not args.force and status.exists():
        print(f"== {image.name}: 処理済みのため飛ばします")
        return
    try:
        match_image(image, args)
        warp_one(image, args)
    except Exception as e:
        traceback.print_exc()
        _write_json(status, {"status": "failed", "error": f"{type(e).__name__}: {e}"})


def cmd_match(args: argparse.Namespace) -> None:
    for image in args.images:
        match_image(image, args)


def cmd_warp(args: argparse.Namespace) -> None:
    for image in args.images:
        warp_one(image, args)


def cmd_run(args: argparse.Namespace) -> None:
    for image in args.images:
        run_one(image, args)


def cmd_batch(args: argparse.Namespace) -> None:
    """ダウンロードと処理を並行して進める。次の図郭を取得しながら、取得済みの図郭を処理する。"""
    res = catalog.filter_resources(catalog.list_resources(args.dataset), args.date, args.mesh)
    total = len(res)
    print(f"{total} 図郭 / 合計 {sum(r.size for r in res) / 1e9:.1f} GB")
    ready: queue.Queue[tuple[int, Path | None]] = queue.Queue(maxsize=args.prefetch)

    def downloader() -> None:
        for i, r in enumerate(res, 1):
            try:
                ready.put((i, catalog.download(r, args.raw)))
            except Exception as e:
                print(f"[{i}/{total}] {r.name}: ダウンロード失敗 {type(e).__name__}: {e}")
                stem = Path(r.name).stem
                _write_json(
                    args.out / stem / "status.json",
                    {"status": "failed", "error": f"download: {type(e).__name__}: {e}"},
                )
                ready.put((i, None))
        ready.put((0, None))

    threading.Thread(target=downloader, daemon=True).start()
    t0 = time.perf_counter()
    while True:
        i, path = ready.get()
        if i == 0:
            break
        if path is None:
            continue
        print(f"[{i}/{total}] 経過 {(time.perf_counter() - t0) / 60:.0f} 分")
        run_one(path, args)
    report.write_report(args.out)
    print(f"完了: {(time.perf_counter() - t0) / 60:.0f} 分。一覧は {args.out / 'report.md'}")


def cmd_report(args: argparse.Namespace) -> None:
    report.write_report(args.out)
    print((args.out / "report.md").read_text("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(prog="satcoreg", description="衛星画像を地理院写真に合わせ込む")
    sub = p.add_subparsers(required=True)

    def add_catalog_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", default=catalog.DEFAULT_DATASET)
        sp.add_argument("--date", nargs="*", help="撮像日 (例: 20261001)")
        sp.add_argument("--mesh", nargs="*", help="図郭番号 (例: 5339K4)")

    def add_match_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--work-res", type=float, default=1.0, help="マッチングの解像度 (m)")
        sp.add_argument("--zoom", type=int, default=17, help="参照に使う地理院タイルのズーム")
        sp.add_argument("--spacing", type=int, default=150, help="格子間隔（作業画素）")
        sp.add_argument("--window", type=int, default=256, help="窓サイズ（作業画素）")
        sp.add_argument("--cache", type=Path, default=Path("data/cache/tiles"))
        sp.add_argument("--tile-concurrency", type=int, default=4, help="地理院タイルの同時取得数")

    def add_warp_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--smoothing", type=float, default=0.1, help="TPS の平滑化")
        sp.add_argument(
            "--min-points", type=int, default=50, help="これより採用点が少ない図郭は補正しない"
        )
        sp.add_argument("--compress", default="DEFLATE", help="COG の圧縮 (DEFLATE / JPEG など)")
        sp.add_argument("--no-verify", dest="verify", action="store_false")

    sp = sub.add_parser("list", help="データセットの GeoTIFF 一覧")
    add_catalog_args(sp)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("download", help="GeoTIFF をダウンロード")
    add_catalog_args(sp)
    sp.add_argument("--out", type=Path, default=Path("data/raw"))
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("match", help="タイポイントを抽出")
    sp.add_argument("images", nargs="+", type=Path)
    sp.add_argument("--out", type=Path, default=Path("data/out"))
    add_match_args(sp)
    sp.set_defaults(func=cmd_match)

    sp = sub.add_parser("warp", help="タイポイントで補正して COG を出力")
    sp.add_argument("images", nargs="+", type=Path)
    sp.add_argument("--out", type=Path, default=Path("data/out"))
    sp.add_argument("--tiepoints", type=Path, help="既定は <out>/<画像名>/tiepoints.geojson")
    add_match_args(sp)
    add_warp_args(sp)
    sp.set_defaults(func=cmd_warp)

    sp = sub.add_parser("run", help="match と warp を続けて実行（処理済みは飛ばす）")
    sp.add_argument("images", nargs="+", type=Path)
    sp.add_argument("--out", type=Path, default=Path("data/out"))
    sp.add_argument("--force", action="store_true", help="処理済みでもやり直す")
    add_match_args(sp)
    add_warp_args(sp)
    sp.set_defaults(func=cmd_run, tiepoints=None)

    sp = sub.add_parser("batch", help="データセットから取得しながら全図郭を処理")
    add_catalog_args(sp)
    sp.add_argument("--raw", type=Path, default=Path("data/raw"))
    sp.add_argument("--out", type=Path, default=Path("data/out"))
    sp.add_argument("--prefetch", type=int, default=2, help="先にダウンロードしておく図郭数")
    sp.add_argument("--force", action="store_true", help="処理済みでもやり直す")
    add_match_args(sp)
    add_warp_args(sp)
    sp.set_defaults(func=cmd_batch, tiepoints=None)

    sp = sub.add_parser("report", help="図郭ごとの結果を一覧にする")
    sp.add_argument("--out", type=Path, default=Path("data/out"))
    sp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)
