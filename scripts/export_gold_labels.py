#!/usr/bin/env python3
"""Yalnızca gold etiketleri tek dosyaya toplar.

Kullanım:
  # Bu makinedeki gold'lar
  python scripts/export_gold_labels.py

  # Arkadaşın gönderdiği klasörü de kat
  python scripts/export_gold_labels.py --labels data/labels --labels /path/arkadas_labels

  # İki export JSON'unu birleştir
  python scripts/export_gold_labels.py --merge data/exports/gold_zehra.json data/exports/gold_arkadas.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iter_label_files(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.glob("*.json")
        if not p.name.endswith("_spec.json") and p.name != "gold_labels.json"
    )


def load_gold(folder: Path, exclude_examples: bool) -> list[dict]:
    gold = []
    for path in iter_label_files(folder):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("status") != "gold":
            continue
        if exclude_examples and data.get("video_id") == "ornek_forklift":
            continue
        data["_source_file"] = path.name
        data["_source_dir"] = str(folder)
        gold.append(data)
    return gold


def spec_view(label: dict) -> dict:
    return {
        "video_id": label.get("video_id"),
        "filename": label.get("filename"),
        "category": label.get("category"),
        "summary": label.get("summary", ""),
        "events": [
            {"time": e.get("time", "00:00"), "event": e.get("event", "")}
            for e in label.get("events", [])
            if e.get("event")
        ],
        "risk": label.get("risk", "Orta"),
        "actions": [a for a in label.get("actions", []) if a],
    }


def merge_by_video_id(items: list[dict]) -> tuple[list[dict], list[dict]]:
    by_id: dict[str, dict] = {}
    conflicts = []
    for item in items:
        key = item.get("video_id") or item.get("filename") or item.get("_source_file")
        if key in by_id:
            prev = by_id[key]
            if json.dumps(spec_view(prev), ensure_ascii=False) != json.dumps(
                spec_view(item), ensure_ascii=False
            ):
                conflicts.append({"video_id": key, "kept": item.get("_source_dir"), "dropped": prev.get("_source_dir")})
        by_id[key] = item
    return list(by_id.values()), conflicts


def write_outputs(labels: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{stem}.json"
    spec_path = out_dir / f"{stem}_spec.json"
    jsonl_path = out_dir / f"{stem}.jsonl"
    csv_path = out_dir / f"{stem}_ozet.csv"

    labels = sorted(labels, key=lambda x: (x.get("category") or "", x.get("filename") or ""))
    specs = [spec_view(x) for x in labels]

    full_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    spec_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in specs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "filename", "risk", "n_events", "summary"])
        for row in labels:
            writer.writerow(
                [
                    row.get("category"),
                    row.get("filename"),
                    row.get("risk"),
                    len(row.get("events") or []),
                    (row.get("summary") or "").replace("\n", " "),
                ]
            )

    counts = Counter(x.get("category") for x in labels)
    print(f"Gold video: {len(labels)}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print(f"Yazıldı:\n  {full_path}\n  {spec_path}\n  {jsonl_path}\n  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", action="append", type=Path, default=[], help="Gold JSON klasörü (birden fazla verilebilir)")
    parser.add_argument("--merge", nargs="+", type=Path, help="Önceki gold_*.json export dosyalarını birleştir")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "exports")
    parser.add_argument("--name", default="gold_labels")
    parser.add_argument("--keep-example", action="store_true")
    args = parser.parse_args()

    collected: list[dict] = []
    if args.merge:
        for path in args.merge:
            collected.extend(json.loads(path.read_text(encoding="utf-8")))
    else:
        folders = args.labels or [ROOT / "data" / "labels"]
        for folder in folders:
            collected.extend(load_gold(folder, exclude_examples=not args.keep_example))

    labels, conflicts = merge_by_video_id(collected)
    if conflicts:
        print(f"Uyarı: {len(conflicts)} video iki kaynakta da gold. Son okunan tutuldu.")
        for c in conflicts:
            print(f"  {c['video_id']}")
    write_outputs(labels, args.out_dir, args.name)


if __name__ == "__main__":
    main()
