#!/usr/bin/env python3
"""15–20 gold videoluk geniş KPI (dengeli kategori).

Örnek:
  python scripts/run_kpi_wide.py --n 18
  python scripts/run_kpi_wide.py --n 18 --model llama3.2-vision:11b

Gold'a dokunmaz. Tahminler data/predictions_wide/ altına gider.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from auto_label_qwen import build_client, competition_view, label_video
from extract_frames import extract_video, safe_id
from run_kpi_sample import SAMPLE_NAMES, predict_one

GOLD_PATH = ROOT / "data" / "exports" / "gold_labels_hepsi.json"


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 9999.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 9999.0


def pick_balanced(
    gold_rows: list[dict],
    videos: dict[str, Path],
    n: int,
    seed: int,
    must_include: list[str],
) -> list[dict]:
    """Her kategoriden mümkün olduğunca eşit; kısa klipler tercih; must_include sabit."""
    by_cat: dict[str, list[dict]] = {"accident": [], "near_miss": [], "normal": []}
    for row in gold_rows:
        cat = row.get("category")
        name = row.get("filename")
        if cat not in by_cat or not name or name not in videos:
            continue
        by_cat[cat].append(row)

    for cat, rows in by_cat.items():
        rows.sort(key=lambda r: (video_duration(videos[r["filename"]]), r["filename"]))

    # 6→18 gibi: önce eşit pay, kalanı en bol kategoriden
    cats = ["accident", "near_miss", "normal"]
    base = n // 3
    extra = n - base * 3
    quota = {c: base for c in cats}
    for i in range(extra):
        quota[cats[i % 3]] += 1

    chosen_names: set[str] = set()
    chosen: list[dict] = []

    # Önce önceki 6'lık sınavı kilitle (karşılaştırma için)
    gold_by_name = {g["filename"]: g for g in gold_rows}
    for name in must_include:
        row = gold_by_name.get(name)
        if row and name in videos and name not in chosen_names:
            chosen.append(row)
            chosen_names.add(name)

    rng = random.Random(seed)
    for cat in cats:
        need = quota[cat]
        already = sum(1 for r in chosen if r.get("category") == cat)
        pool = [r for r in by_cat[cat] if r["filename"] not in chosen_names]
        # Kısa kliplerin ilk yarısından rastgele doldur (hız + çeşitlilik)
        head = pool[: max(need * 3, need)]
        rng.shuffle(head)
        for row in head:
            if already >= need:
                break
            chosen.append(row)
            chosen_names.add(row["filename"])
            already += 1
        # yetmezse kalan havuz
        if already < need:
            for row in pool:
                if already >= need:
                    break
                if row["filename"] in chosen_names:
                    continue
                chosen.append(row)
                chosen_names.add(row["filename"])
                already += 1

    # Kategori sırası: accident → near_miss → normal
    order = {"accident": 0, "near_miss": 1, "normal": 2}
    chosen.sort(key=lambda r: (order.get(r.get("category"), 9), r.get("filename") or ""))
    return chosen[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=18, help="Video sayısı (15–20 önerilir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
        help="Ollama model adı",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "data" / "predictions_wide",
    )
    parser.add_argument(
        "--no-second-look",
        action="store_true",
        help="İkinci bakışı kapat (daha hızlı, biraz daha zayıf)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = modele göre (qwen 8, llava 3)")
    parser.add_argument(
        "--every-sec",
        type=float,
        default=0.5,
        help="Eşit aralıklı örnekleme (hareket yoksa); denser = daha iyi zaman KPI",
    )
    args = parser.parse_args()
    if args.n < 6:
        raise SystemExit("--n en az 6 olmalı")

    os.environ["OLLAMA_MODEL"] = args.model
    safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.model)
    max_frames = args.max_frames or (3 if "llava" in args.model.lower() else 8)

    gold_all = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    videos = {
        p.name: p
        for p in (ROOT / "data" / "videos").rglob("*")
        if p.suffix.lower() == ".mp4"
    }
    sample = pick_balanced(gold_all, videos, args.n, args.seed, SAMPLE_NAMES)
    if len(sample) < args.n:
        print(f"Uyarı: istenen {args.n}, seçilen {len(sample)} (dosya/kategori yetmez)")

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    client, model = build_client("ollama")
    print(f"Geniş KPI  model={model}  n={len(sample)}  seed={args.seed}")
    from collections import Counter

    print("Dağılım:", dict(Counter(r.get("category") for r in sample)))

    # Seçim listesini kaydet (tekrarlanabilir)
    list_path = ROOT / "data" / "exports" / "kpi_wide_selection.json"
    list_path.write_text(
        json.dumps(
            [{"filename": r["filename"], "category": r.get("category")} for r in sample],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Seçim listesi: {list_path}")

    ok_gold: list[dict] = []
    for i, gold in enumerate(sample, start=1):
        name = gold["filename"]
        video = videos.get(name)
        print(f"\n[{i}/{len(sample)}] [{gold.get('category')}] {name}")
        if not video:
            print("  ATLANDI (video yok)")
            continue
        try:
            # predict_one içinde second look açık; hızlı mod için monkeypatch
            if args.no_second_look:
                from utils.scene_evidence import analyze_video

                evidence = analyze_video(video)
                frames_meta = extract_video(
                    video,
                    ROOT / "data/frames",
                    every_sec=args.every_sec,
                    max_frames=max_frames,
                    use_motion=True,
                )
                label = label_video(
                    client,
                    model,
                    video,
                    frames_meta,
                    use_folder_hint=False,
                    backend="ollama",
                    evidence=evidence,
                    use_second_look=False,
                )
            else:
                # predict_one sabit 6 kare kullanır; LLaVA / denser için yerel yol
                if max_frames != 6:
                    from utils.scene_evidence import analyze_video

                    evidence = analyze_video(video)
                    frames_meta = extract_video(
                        video,
                        ROOT / "data/frames",
                        every_sec=args.every_sec,
                        max_frames=max_frames,
                        use_motion=True,
                    )
                    label = label_video(
                        client,
                        model,
                        video,
                        frames_meta,
                        use_folder_hint=False,
                        backend="ollama",
                        evidence=evidence,
                        use_second_look=True,
                    )
                else:
                    label = predict_one(client, model, video)
        except Exception as exc:
            print(f"  HATA: {exc}")
            continue
        ok_gold.append(gold)
        vid = safe_id(video)
        (args.pred_dir / f"{vid}.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.pred_dir / f"{vid}_spec.json").write_text(
            json.dumps(competition_view(label), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  tahmin risk={label.get('risk')}  cat={label.get('category')}  "
            f"olay={len(label.get('events') or [])}"
        )

    gold_out = ROOT / "data" / "exports" / f"kpi_wide_{safe_model}_gold.json"
    gold_out.write_text(json.dumps(ok_gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGold alt küme: {gold_out} ({len(ok_gold)} video)")

    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_kpi.py"),
                "--gold",
                str(gold_out),
                "--pred",
                str(args.pred_dir),
                "--out-json",
                str(ROOT / "data" / "exports" / f"kpi_wide_{safe_model}_report.json"),
                "--out-csv",
                str(ROOT / "data" / "exports" / f"kpi_wide_{safe_model}_ozet.csv"),
            ]
        )
    )


if __name__ == "__main__":
    main()
