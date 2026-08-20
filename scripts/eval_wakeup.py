#!/usr/bin/env python3
"""Wake-up katmanı kalibrasyonu: kare seçimimiz gold olay anını yakalıyor mu?

Model çağırmaz, disk yazmaz — sadece hareket profilini ve seçilen kare zamanlarını
hesaplar. Neden önemli: olay zamanları kare zamanlarına yapıştığı için, gold olayın
±2 saniyesinde hiç kare seçilmediyse o olayı yakalamamız **matematiksel olarak
imkânsız**. Yani buradaki "kapsama" oranı, olay yakalama metriğinin üst sınırı.

Ayrıca hareket temelli seçimi eşit aralıklı (naif) seçimle karşılaştırır; wake-up
katmanının katkısını sayı olarak verir.

    python scripts/eval_wakeup.py                        # varsayılan 8 kare
    python scripts/eval_wakeup.py --grid 6,8,10,12       # kare sayısı taraması
    python scripts/eval_wakeup.py --tolerance 2 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import motion_scores, pick_times, sample_times  # noqa: E402

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.kpi import normalize_ambiguity  # noqa: E402
from utils.spec_output import mmss_to_seconds  # noqa: E402

GOLD_DEFAULT = ROOT / "data" / "exports" / "gold_labels_hepsi.json"
VIDEO_ROOT = ROOT / "data" / "videos"
OUT_CSV = ROOT / "data" / "exports" / "wakeup_kalibrasyon.csv"


def video_index() -> dict[str, Path]:
    return {path.name: path for path in VIDEO_ROOT.rglob("*") if path.is_file()}


def profile(video: Path) -> tuple[float, list[tuple[float, float]]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.0, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frames / fps if fps else 0.0
    motion = motion_scores(cap, fps)
    cap.release()
    return duration, motion


def covered(event_sec: float, times: list[float], tol: float) -> bool:
    return any(abs(t - event_sec) <= tol for t in times)


def _top_peaks(motion: list[tuple[float, float]], count: int, min_gap: float) -> list[float]:
    """En yüksek skorlu, birbirinden min_gap saniye uzak tepeler."""
    picked: list[float] = []
    for t, score in sorted(motion, key=lambda item: item[1], reverse=True):
        if score <= 0:
            continue
        if all(abs(t - other) >= min_gap for other in picked):
            picked.append(t)
        if len(picked) >= count:
            break
    return picked


def main() -> None:  # noqa: C901 - rapor üretimi tek akışta okunur
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Uzun video wake-up penceresinin yarı genişliği (utils.demo_pipeline.WAKE_WINDOW_S)",
    )
    parser.add_argument("--grid", default="8", help="Kare sayıları, virgülle: 6,8,10")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    grid = [int(x) for x in args.grid.split(",") if x.strip()]
    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        gold_rows = gold_rows[: args.limit]
    videos = video_index()

    rows: list[dict[str, Any]] = []
    totals: dict[int, list[int]] = {mf: [0, 0] for mf in grid}  # [hit, total]
    naive: dict[int, list[int]] = {mf: [0, 0] for mf in grid}
    peak_hit = [0, 0]
    top3_hit = [0, 0]
    window_hit = [0, 0]
    missing = 0

    print(f"{len(gold_rows)} gold video taranıyor (model yok, disk yazımı yok)...")
    for row in gold_rows:
        name = str(row.get("filename") or "")
        events = [e for e in (row.get("events") or []) if e.get("event")]
        video = videos.get(name)
        if video is None or not events:
            missing += 1
            continue

        duration, motion = profile(video)
        if duration <= 0:
            missing += 1
            continue
        peak_t = max(motion, key=lambda item: item[1])[0] if motion else None
        top3 = _top_peaks(motion, 3, min_gap=2.0)

        for event in events:
            event_sec = float(mmss_to_seconds(event.get("time")))
            record: dict[str, Any] = {
                "filename": name,
                "category": row.get("category"),
                "ambiguity": normalize_ambiguity(row.get("ambiguity")),
                "gold_time": event.get("time"),
                "sure_sn": round(duration, 1),
                "hareket_tepesi_sn": round(peak_t, 1) if peak_t is not None else "",
            }
            if peak_t is not None:
                peak_hit[1] += 1
                if abs(peak_t - event_sec) <= args.tolerance:
                    peak_hit[0] += 1
                record["tepe_farki_sn"] = round(abs(peak_t - event_sec), 1)

                # Uzun video yolu: kareler tepenin ±WAKE_WINDOW_S'i içinden seçiliyor.
                # Olay o pencerenin dışında kalırsa uzun videoda olayı hiç görmeyiz.
                window_hit[1] += 1
                if abs(peak_t - event_sec) <= args.window:
                    window_hit[0] += 1
                record["pencere_ici"] = "evet" if abs(peak_t - event_sec) <= args.window else "hayır"

            if top3:
                top3_hit[1] += 1
                if any(abs(t - event_sec) <= args.tolerance for t in top3):
                    top3_hit[0] += 1

            for max_frames in grid:
                wake_times = pick_times(duration, max_frames, motion)
                flat_times = sample_times(duration, 0.75, max_frames)
                hit = covered(event_sec, wake_times, args.tolerance)
                totals[max_frames][1] += 1
                totals[max_frames][0] += int(hit)
                naive[max_frames][1] += 1
                naive[max_frames][0] += int(covered(event_sec, flat_times, args.tolerance))
                record[f"kapsam_{max_frames}"] = "evet" if hit else "hayır"
                if max_frames == grid[0]:
                    record["secilen_zamanlar"] = ",".join(f"{t:.0f}" for t in wake_times)
            rows.append(record)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 66)
    print("WAKE-UP KALİBRASYONU")
    print("=" * 66)
    print(f"  Ölçülen gold olay: {len(rows)}  |  atlanan video: {missing}")
    print(f"  Tolerans: ±{args.tolerance:.0f} sn\n")
    print(f"  {'kare':>5}  {'hareketli seçim':>16}  {'eşit aralıklı':>14}  fark")
    for max_frames in grid:
        hit, total = totals[max_frames]
        n_hit, n_total = naive[max_frames]
        wake_rate = hit / max(total, 1)
        flat_rate = n_hit / max(n_total, 1)
        print(
            f"  {max_frames:>5}  {wake_rate:>15.0%}  {flat_rate:>13.0%}  "
            f"{wake_rate - flat_rate:+.0%}"
        )
    if peak_hit[1]:
        print(
            f"\n  En yüksek hareket tepesi gold olayın ±{args.tolerance:.0f} sn içinde: "
            f"{peak_hit[0] / peak_hit[1]:.0%}"
        )
    if top3_hit[1]:
        print(
            f"  İlk 3 hareket tepesinden biri ±{args.tolerance:.0f} sn içinde: "
            f"{top3_hit[0] / top3_hit[1]:.0%}"
        )
    if window_hit[1]:
        print(
            f"  Olay, tepe ±{args.window:.0f} sn penceresinin içinde "
            f"(uzun video yolu): {window_hit[0] / window_hit[1]:.0%}"
        )

    best = max(grid, key=lambda mf: totals[mf][0] / max(totals[mf][1], 1))
    ceiling = totals[best][0] / max(totals[best][1], 1)
    print(
        f"\n  Olay yakalama üst sınırı ({best} kare ile): {ceiling:.0%}. "
        "Metrik bunun üstüne çıkamaz; kalan kayıp model ve dil tarafındadır."
    )

    by_cat: dict[str, list[int]] = {}
    for record in rows:
        key = f"{record['category']}/{record['ambiguity']}"
        bucket = by_cat.setdefault(key, [0, 0])
        bucket[1] += 1
        bucket[0] += int(record[f"kapsam_{best}"] == "evet")
    print("\n  Kırılım (kategori/belirsizlik):")
    for key, (hit, total) in sorted(by_cat.items()):
        print(f"    {key:22} {hit}/{total}  {hit / max(total, 1):.0%}")

    print(f"\nSatır satır tablo: {args.out_csv}")


if __name__ == "__main__":
    main()
