#!/usr/bin/env python3
"""Gold videoluk geniş KPI (dengeli örnek veya tüm gold).

Örnek:
  python scripts/run_kpi_wide.py --n 18
  python scripts/run_kpi_wide.py --n all
  python scripts/run_kpi_wide.py --n all --split holdout --holdout-frac 0.2
  python scripts/run_kpi_wide.py --n all --split dev      # splits.json (ayar kümesi)

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
from utils.splits import filter_by_split

GOLD_PATH = ROOT / "data" / "exports" / "gold_labels_hepsi.json"


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 9999.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 9999.0


def available_gold(
    gold_rows: list[dict],
    videos: dict[str, Path],
) -> list[dict]:
    out = []
    for row in gold_rows:
        name = row.get("filename")
        if name and name in videos and row.get("category") in {
            "accident",
            "near_miss",
            "normal",
        }:
            out.append(row)
    order = {"accident": 0, "near_miss": 1, "normal": 2}
    out.sort(key=lambda r: (order.get(r.get("category"), 9), r.get("filename") or ""))
    return out


def split_holdout(
    rows: list[dict],
    seed: int,
    holdout_frac: float,
    which: str,
) -> list[dict]:
    """Kategori içinde seed'li holdout; which=all|train|holdout."""
    if which == "all":
        return list(rows)
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {"accident": [], "near_miss": [], "normal": []}
    for row in rows:
        by_cat[row["category"]].append(row)
    train: list[dict] = []
    hold: list[dict] = []
    for _cat, pool in by_cat.items():
        pool = list(pool)
        rng.shuffle(pool)
        n_hold = max(1, int(round(len(pool) * holdout_frac))) if pool else 0
        if len(pool) <= 2:
            n_hold = 1 if len(pool) == 2 else 0
        hold.extend(pool[:n_hold])
        train.extend(pool[n_hold:])
    order = {"accident": 0, "near_miss": 1, "normal": 2}
    chosen = hold if which == "holdout" else train
    chosen.sort(key=lambda r: (order.get(r.get("category"), 9), r.get("filename") or ""))
    return chosen


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

    cats = ["accident", "near_miss", "normal"]
    base = n // 3
    extra = n - base * 3
    quota = {c: base for c in cats}
    for i in range(extra):
        quota[cats[i % 3]] += 1

    chosen_names: set[str] = set()
    chosen: list[dict] = []

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
        head = pool[: max(need * 3, need)]
        rng.shuffle(head)
        for row in head:
            if already >= need:
                break
            chosen.append(row)
            chosen_names.add(row["filename"])
            already += 1
        if already < need:
            for row in pool:
                if already >= need:
                    break
                if row["filename"] in chosen_names:
                    continue
                chosen.append(row)
                chosen_names.add(row["filename"])
                already += 1

    order = {"accident": 0, "near_miss": 1, "normal": 2}
    chosen.sort(key=lambda r: (order.get(r.get("category"), 9), r.get("filename") or ""))
    return chosen[:n]


def parse_n(raw: str) -> int | None:
    """None = tüm uygun gold; int = dengeli örnek boyutu."""
    text = (raw or "18").strip().lower()
    if text in {"all", "full", "*"}:
        return None
    return int(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        default="18",
        help="Video sayısı (varsayılan 18) veya 'all' (tüm gold)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout", "dev", "test"),
        default="all",
        help="all|dev|test = splits.json; train|holdout = eski seed'li ayırım (--n all)",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="--split train|holdout için kategori içi oran",
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "teknofest", "openai"),
        default="ollama",
        help="teknofest = yarışmanın ortak API'si (.env: TEKNOFEST_BASE_URL, VLM_MODEL)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model adı (boş: ollama'da OLLAMA_MODEL, teknofest'te VLM_MODEL)",
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
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Kural katmanını (risk_rules.refine_label) kapat — ablasyon ölçümü",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Çıktı dosya adına ek etiket (örn. norefine, holdout)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = modele göre (qwen 8, llava 3)")
    parser.add_argument(
        "--every-sec",
        type=float,
        default=0.5,
        help="Eşit aralıklı örnekleme (hareket yoksa); denser = daha iyi zaman KPI",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="pred-dir'de mevcut JSON varsa videoyu atla (Ollama crash sonrası devam)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Virgüllü dosya adı parçası; yalnız eşleşen videoları koş (yeniden etiket)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Bitince evaluate_kpi çalıştırma (hedefli yeniden koşu)",
    )
    args = parser.parse_args()
    n_videos = parse_n(str(args.n))
    if n_videos is not None and n_videos < 6 and not str(args.only).strip():
        raise SystemExit("--n en az 6 olmalı (veya --n all / --only)")

    if args.backend == "ollama":
        model_name = args.model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
        os.environ["OLLAMA_MODEL"] = model_name
    else:
        if args.model:
            os.environ["VLM_MODEL"] = args.model
        model_name = args.model or os.getenv("VLM_MODEL", "vlm")
    safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    if args.tag:
        safe_model = f"{safe_model}_{''.join(c if c.isalnum() else '_' for c in args.tag)}"
    max_frames = args.max_frames or (3 if "llava" in model_name.lower() else 8)

    gold_all = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    videos = {
        p.name: p
        for p in (ROOT / "data" / "videos").rglob("*")
        if p.suffix.lower() == ".mp4"
    }
    if n_videos is None:
        pool = available_gold(gold_all, videos)
        if args.split in {"dev", "test"}:
            sample = filter_by_split(pool, args.split)
            mode = f"all/{args.split}"
        else:
            sample = split_holdout(pool, args.seed, args.holdout_frac, args.split)
            mode = f"all/{args.split}"
    else:
        if args.split in {"dev", "test"}:
            pool = filter_by_split(available_gold(gold_all, videos), args.split)
            sample = pool[:n_videos] if n_videos < len(pool) else pool
            mode = f"{args.split}/{len(sample)}"
        else:
            if args.split != "all":
                print("Uyarı: --split train|holdout yalnız --n all ile anlamlı; yok sayıldı.")
            sample = pick_balanced(gold_all, videos, n_videos, args.seed, SAMPLE_NAMES)
            mode = f"balanced/{n_videos}"
            if len(sample) < n_videos:
                print(f"Uyarı: istenen {n_videos}, seçilen {len(sample)} (dosya/kategori yetmez)")

    if str(args.only).strip():
        needles = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        sample = [
            row
            for row in sample
            if any(n in (row.get("filename") or "").lower() for n in needles)
        ]
        mode = f"{mode}+only/{len(sample)}"

    video_root = ROOT / "data" / "videos"
    if not sample:
        n_gold = len(gold_all)
        n_vid = len(videos)
        splits = ROOT / "data" / "exports" / "splits.json"
        raise SystemExit(
            "KPI kümesi boş (n=0). Bu skor değil, veri bağı kopuk.\n"
            f"  gold kayıt : {n_gold}  ({GOLD_PATH})\n"
            f"  diskte mp4 : {n_vid}  ({video_root} symlink={video_root.is_symlink()})\n"
            f"  splits.json: {splits.exists()}  ({splits})\n"
            "Colab'de hücre 3 (git reset) 4a'yı bozar. 4a'yı tekrar çalıştırın, "
            "hücre 5'te 'Drive video: 77' görünce 6a'ya dönün. Ollama kurmanıza gerek yok."
        )

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    client, model = build_client(args.backend)
    print(
        f"Geniş KPI  backend={args.backend}  model={model}  "
        f"n={len(sample)}  seed={args.seed}  mode={mode}"
    )
    from collections import Counter

    print("Dağılım:", dict(Counter(r.get("category") for r in sample)))

    list_path = ROOT / "data" / "exports" / "kpi_wide_selection.json"
    list_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "seed": args.seed,
                "backend": args.backend,
                "model": model_name,
                "videos": [
                    {"filename": r["filename"], "category": r.get("category")}
                    for r in sample
                ],
            },
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
        vid = safe_id(video)
        pred_json = args.pred_dir / f"{vid}.json"
        if args.resume and pred_json.exists():
            ok_gold.append(gold)
            print("  resume: mevcut tahmin var, atlandı")
            continue
        try:
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
                    backend=args.backend,
                    evidence=evidence,
                    use_second_look=False,
                    use_refine=not args.no_refine,
                )
            else:
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
                        backend=args.backend,
                        evidence=evidence,
                        use_second_look=True,
                        use_refine=not args.no_refine,
                    )
                else:
                    label = predict_one(
                        client,
                        model,
                        video,
                        backend=args.backend,
                        use_refine=not args.no_refine,
                    )
        except Exception as exc:
            print(f"  HATA: {exc}")
            continue
        ok_gold.append(gold)
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

    if args.no_eval:
        print(f"\nYeniden etiket bitti: {len(ok_gold)} video  ({args.pred_dir})")
        return

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
