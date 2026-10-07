"""図郭ごとの結果（match / verify / status）を一覧表にまとめる。"""

import csv
import json
from pathlib import Path

from satcoreg.catalog import NAME_RE

MIN_POINTS = 100  # これより採用点が少ない図郭は目視確認に回す
MAX_RESIDUAL_M = 2.0  # 補正後の残差中央値がこれを超える図郭は目視確認に回す
MAX_CLOUD = 0.5  # 雲の割合がこれを超える図郭は目視確認に回す

COLUMNS = [
    ("mesh", "図郭"),
    ("date", "撮像日"),
    ("status", "状態"),
    ("cloud", "雲の割合"),
    ("candidates", "候補点"),
    ("ok_before", "採用点"),
    ("shift_median", "補正前 中央値 (m)"),
    ("shift_p90", "補正前 90% (m)"),
    ("resid_median", "補正後 中央値 (m)"),
    ("resid_p90", "補正後 90% (m)"),
    ("check", "要確認"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def collect(out_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        m = NAME_RE.match(d.name + ".tif")
        if not m:
            continue
        match = _load(d / "match_summary.json")
        verify = _load(d / "verify_summary.json")
        status = _load(d / "status.json")
        row = {
            "mesh": m["mesh"].upper(),
            "date": m["date"],
            "status": status.get("status", "running" if match else "pending"),
            "cloud": match.get("cloud_frac", ""),
            "candidates": match.get("candidates", ""),
            "ok_before": match.get("status", {}).get("ok", ""),
            "shift_median": match.get("shift_m_median", ""),
            "shift_p90": match.get("shift_m_p90", ""),
            "resid_median": verify.get("shift_m_median", ""),
            "resid_p90": verify.get("shift_m_p90", ""),
        }
        reasons = []
        if row["status"] != "done":
            reasons.append(status.get("reason") or status.get("error") or row["status"])
        if isinstance(row["ok_before"], int) and row["ok_before"] < MIN_POINTS:
            reasons.append(f"採用点 {row['ok_before']}")
        if isinstance(row["cloud"], float) and row["cloud"] > MAX_CLOUD:
            reasons.append(f"雲 {row['cloud']:.0%}")
        if isinstance(row["resid_median"], float) and row["resid_median"] > MAX_RESIDUAL_M:
            reasons.append(f"残差 {row['resid_median']} m")
        row["check"] = " / ".join(reasons)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["date"], r["mesh"]))


def write_report(out_dir: Path) -> list[dict]:
    rows = collect(out_dir)
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([label for _, label in COLUMNS])
        w.writerows([[r[k] for k, _ in COLUMNS] for r in rows])

    done = [r for r in rows if r["status"] == "done"]
    lines = [
        f"処理済み {len(done)} / {len(rows)} 図郭、要確認 {sum(bool(r['check']) for r in rows)} 図郭",
        "",
        "| " + " | ".join(label for _, label in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    lines += ["| " + " | ".join(str(r[k]) for k, _ in COLUMNS) + " |" for r in rows]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return rows
