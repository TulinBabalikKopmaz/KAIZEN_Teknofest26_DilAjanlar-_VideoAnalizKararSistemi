#!/usr/bin/env python3
"""Demo provası: 3 prompt varyantı x N video koşup süre ve çıktı kalitesini raporlar.

Jüri hangi promptu verirse versin sistemin 60 saniyede anlamlı cevap ürettiğini
görmek için. Sunum öncesi son kontrol.

    python scripts/demo_rehearsal.py --videos data/videos --limit 3
    python scripts/demo_rehearsal.py --videos demo1.mp4 demo2.mp4 --fast

Çıktı: data/exports/demo_rehearsal.csv + ekrana özet ve kontrol listesi.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import VIDEO_EXTS, safe_id  # noqa: E402

from utils import config  # noqa: E402
from utils.demo_pipeline import run_demo_analysis, use_utf8_stdout  # noqa: E402

PROMPT_VARIANTS: tuple[str, ...] = (
    "Bu videoda bir iş kazası var mı? Varsa ne olduğunu ve kaçıncı saniyede olduğunu söyle.",
    "Videoda iş güvenliği ihlali veya tehlikeli davranış var mı? Zaman damgasıyla açıkla "
    "ve alınması gereken önlemleri yaz.",
    "Sahada ramak kala bir olay yaşandı mı? Yaşandıysa saniyesini, sebebini ve acil "
    "müdahale adımlarını raporla.",
)

OUT_CSV = ROOT / "data" / "exports" / "demo_rehearsal.csv"
FIELDS = [
    "video",
    "prompt_no",
    "total_s",
    "risk",
    "category",
    "event_count",
    "first_event",
    "answer_chars",
    "fallback_used",
    "frames",
    "warnings",
]

CHECKLIST = (
    "Video formatları: mp4, mov, avi ve dikey (9:16) klip ile en az bir kez dene.",
    "Uzun video: 3+ dakikalık kayıtta wake-up penceresi doğru yeri buluyor mu?",
    "Ağ kesintisi: ortak API'yi kapatıp PROVIDER=ollama yedeğine düşüşü prova et.",
    "Soğuk başlangıç: ilk çağrı yavaş; demodan 5 dk önce bir kez ısıtma koşusu yap.",
    "Ekran: Streamlit'i demo çözünürlüğünde aç, tek ekranda kaydırma olmasın.",
    "Yedek plan: son başarılı data/demo_runs klasörünü açık bir sekmede hazır tut.",
    "Süre: hedef 40 sn; aşarsa DEMO_FAST_MODE=1 veya --max-frames 6 ile koş.",
)


def collect_videos(inputs: list[Path], limit: int) -> list[Path]:
    videos: list[Path] = []
    for item in inputs:
        if item.is_dir():
            videos.extend(p for p in sorted(item.rglob("*")) if p.suffix.lower() in VIDEO_EXTS)
        elif item.suffix.lower() in VIDEO_EXTS:
            videos.append(item)
    unique: list[Path] = []
    for video in videos:
        if video not in unique:
            unique.append(video)
    return unique[:limit] if limit else unique


async def rehearse(videos: list[Path], prompts: tuple[str, ...], **kwargs) -> list[dict]:
    rows: list[dict] = []
    total = len(videos) * len(prompts)
    step = 0
    for video in videos:
        for index, prompt in enumerate(prompts, start=1):
            step += 1
            print(f"\n=== [{step}/{total}] {video.name} | prompt #{index} ===")
            try:
                result = await run_demo_analysis(
                    video,
                    prompt,
                    run_name=f"{safe_id(video)}__p{index}",
                    **kwargs,
                )
            except Exception as exc:
                print(f"  HATA: {exc}")
                rows.append(
                    {
                        "video": video.name,
                        "prompt_no": index,
                        "total_s": "",
                        "risk": "HATA",
                        "category": "",
                        "event_count": 0,
                        "first_event": "",
                        "answer_chars": 0,
                        "fallback_used": "",
                        "frames": 0,
                        "warnings": str(exc)[:200],
                    }
                )
                continue

            events = result.spec.get("events") or []
            rows.append(
                {
                    "video": video.name,
                    "prompt_no": index,
                    "total_s": round(result.total_s, 1),
                    "risk": result.spec.get("risk"),
                    "category": result.label.get("category"),
                    "event_count": len(events),
                    "first_event": events[0].get("time") if events else "",
                    "answer_chars": len(result.answer),
                    "fallback_used": any(call.get("fallback") for call in result.model_calls),
                    "frames": len(result.frames),
                    "warnings": " | ".join(result.warnings)[:200],
                }
            )
    return rows


def report(rows: list[dict]) -> None:
    durations = [row["total_s"] for row in rows if isinstance(row["total_s"], (int, float))]
    failures = [row for row in rows if row["risk"] == "HATA"]
    thin_answers = [row for row in rows if row["answer_chars"] and row["answer_chars"] < 40]
    no_events = [row for row in rows if row["risk"] != "HATA" and row["event_count"] == 0]
    over_budget = [d for d in durations if d > 40.0]
    over_limit = [d for d in durations if d > 60.0]

    print("\n" + "=" * 60)
    print("PROVA ÖZETİ")
    print("=" * 60)
    print(f"  Koşu sayısı        : {len(rows)}  (hata: {len(failures)})")
    if durations:
        print(f"  Süre medyan / maks : {statistics.median(durations):.1f} / {max(durations):.1f} sn")
        print(f"  40 sn üstü         : {len(over_budget)}")
        print(f"  60 sn üstü (RİSK)  : {len(over_limit)}")
    print(f"  Kısa/boş cevap     : {len(thin_answers)}")
    print(f"  Olay çıkmayan koşu : {len(no_events)}")
    fallbacks = [row for row in rows if row.get("fallback_used") is True]
    print(f"  Yedek sağlayıcıya düşen: {len(fallbacks)}")

    print("\nDEMO ÖNCESİ KONTROL LİSTESİ")
    for i, item in enumerate(CHECKLIST, start=1):
        print(f"  {i}. {item}")


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "data" / "videos"],
        help="Video dosyaları veya klasör",
    )
    parser.add_argument("--limit", type=int, default=3, help="Kaç video (0 = hepsi)")
    parser.add_argument("--provider", choices=config.PROVIDERS, default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Her satırı bir prompt olan dosya (varsayılan: 3 dahili varyant)",
    )
    args = parser.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    videos = collect_videos(args.videos, args.limit)
    if not videos:
        raise SystemExit(f"Video bulunamadı: {[str(v) for v in args.videos]}")

    prompts = PROMPT_VARIANTS
    if args.prompts_file:
        lines = [
            line.strip()
            for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = tuple(lines) or PROMPT_VARIANTS

    print(config.describe())
    print(f"{len(videos)} video x {len(prompts)} prompt = {len(videos) * len(prompts)} koşu")

    rows = asyncio.run(
        rehearse(
            videos,
            prompts,
            fast=args.fast or None,
            max_frames=args.max_frames or None,
            use_rag=not args.no_rag,
        )
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    print(f"\nTablo: {args.out_csv}")

    report(rows)


if __name__ == "__main__":
    main()
