#!/usr/bin/env python3
"""6 gold videoluk gerçek KPI (2 kaza, 2 near miss, 2 normal).

Gold dosyaların üzerine yazmaz. Tahminler data/predictions/ altına gider.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from auto_label_qwen import build_client, competition_view, label_video
from extract_frames import extract_video, safe_id

SAMPLE_NAMES = [
    "forklift guy was chilling when a woman attacks his forks [s-UHzuOh3B4].mp4",
    "2uuwpx1ykPQ_trim_0.mp4",
    "How NOT to UNLOAD a truck [t0abC-2oucI].mp4",
    "Near miss walk into the path of a forklift [EHyWjeJi1Oc].mp4",
    "clip_TAR_Plant_DC-Floor_left_20251115_000000_0000.mp4",
    "clip_Tippler_1&2_CP_IP_Cam_20251127_000108_0002.mp4",
]


def predict_one(client, model: str, video: Path) -> dict:
    # 1) Önce sensör kanıtı (YOLO / hareket / yangın rengi) — Qwen'den bağımsız
    from utils.scene_evidence import analyze_video

    evidence = analyze_video(video)
    print(
        "  kanıt:"
        f" hareket={'evet' if evidence.motion_elevated else 'hayır'}"
        f" yolo={'var' if evidence.yolo_available else 'yok'}"
        f" yakın={evidence.person_vehicle_close}"
        f" çok_yakın={evidence.person_vehicle_very_close}"
        f" yangın_şüphe={evidence.fire_suspect}"
    )

    frames_meta = extract_video(
        video,
        ROOT / "data/frames",
        every_sec=0.75,
        max_frames=6,
        use_motion=True,
    )
    times = [f["time"] for f in frames_meta["frames"]]
    print(f"  kareler={times}")
    try:
        return label_video(
            client,
            model,
            video,
            frames_meta,
            use_folder_hint=False,
            backend="ollama",
            evidence=evidence,
            use_second_look=True,
        )
    except Exception as exc:
        print(f"  6 kare başarısız ({exc}); 4 kare ile tekrar")
        frames_meta = extract_video(
            video,
            ROOT / "data/frames",
            every_sec=1.0,
            max_frames=4,
            use_motion=True,
        )
        return label_video(
            client,
            model,
            video,
            frames_meta,
            use_folder_hint=False,
            backend="ollama",
            evidence=evidence,
            use_second_look=True,
        )


def main() -> None:
    gold_all = json.loads((ROOT / "data/exports/gold_labels_hepsi.json").read_text(encoding="utf-8"))
    gold_by_name = {g["filename"]: g for g in gold_all}
    videos = {
        p.name: p
        for p in (ROOT / "data/videos").rglob("*")
        if p.suffix.lower() == ".mp4"
    }

    pred_dir = ROOT / "data/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sample_gold = []

    client, model = build_client("ollama")
    print(f"Qwen KPI örneği  model={model}  n={len(SAMPLE_NAMES)}")

    for name in SAMPLE_NAMES:
        gold = gold_by_name.get(name)
        video = videos.get(name)
        if not gold or not video:
            print(f"ATLANDI (dosya yok): {name}")
            continue
        sample_gold.append(gold)
        print(f"\n[{gold.get('category')}] {name}")
        try:
            label = predict_one(client, model, video)
        except Exception as exc:
            print(f"  HATA, bu video atlandı: {exc}")
            continue
        vid = safe_id(video)
        (pred_dir / f"{vid}.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pred_dir / f"{vid}_spec.json").write_text(
            json.dumps(competition_view(label), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  tahmin risk={label.get('risk')}  olay={len(label.get('events') or [])}")

    sample_path = ROOT / "data/exports/kpi_sample_gold.json"
    sample_path.write_text(json.dumps(sample_gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nÖrnek gold: {sample_path} ({len(sample_gold)} video)")

    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_kpi.py"),
                "--gold",
                str(sample_path),
                "--pred",
                str(pred_dir),
                "--out-json",
                str(ROOT / "data/exports/kpi_sample_report.json"),
                "--out-csv",
                str(ROOT / "data/exports/kpi_sample_ozet.csv"),
            ]
        )
    )


if __name__ == "__main__":
    main()
