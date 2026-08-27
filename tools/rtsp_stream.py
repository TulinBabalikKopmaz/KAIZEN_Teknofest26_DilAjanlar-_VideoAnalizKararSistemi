#!/usr/bin/env python3
"""Gerçek zamanlı akış CLI: RTSP / webcam / dosya → wake-up tetikli analiz.

Gövde `utils/live_watch.py`. Jüri demo ekranı değil; sunumda operatör konsolu
`streamlit run app/live_app.py`.

    python tools/rtsp_stream.py --source data/videos/ornek.mp4 --loop
    python tools/rtsp_stream.py --source 0
    python tools/rtsp_stream.py --source rtsp://kullanici:sifre@10.0.0.12:554/stream1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import strftime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import config  # noqa: E402
from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.live_watch import ALERTS_DIR, StreamConfig, run  # noqa: E402


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="RTSP URL, kamera indeksi (0) veya video dosyası")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--motion-threshold", type=float, default=12.0)
    parser.add_argument("--cooldown", type=float, default=8.0)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--pre-frames", type=int, default=6)
    parser.add_argument("--vlm-frames", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1, help="Eşzamanlı VLM analizi sayısı")
    parser.add_argument("--duration", type=float, default=0.0, help="0 = akış bitene / Ctrl+C'ye kadar")
    parser.add_argument("--loop", action="store_true", help="Dosya kaynağını başa sar (canlı taklidi)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = StreamConfig(
        source=args.source,
        sample_fps=args.sample_fps,
        motion_threshold=args.motion_threshold,
        cooldown_s=args.cooldown,
        clip_frames=args.clip_frames,
        pre_frames=args.pre_frames,
        vlm_frames=max(args.vlm_frames, 4),
        max_workers=max(args.workers, 1),
        loop_file=args.loop,
        save_alerts=not args.no_save,
    )

    endpoint = config.vlm_endpoint()
    print(f"VLM: {endpoint.provider} / {endpoint.model}")
    try:
        alerts = asyncio.run(run(cfg, args.duration))
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return

    print("\n" + "=" * 60)
    print(f"Toplam uyarı: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert['trigger_time']}  {alert['spec'].get('risk')}  {alert['spec'].get('summary', '')[:70]}")
    if alerts:
        latencies = [alert["latency_s"] for alert in alerts]
        print(f"  Analiz gecikmesi: ortalama {sum(latencies) / len(latencies):.1f} sn, en kötü {max(latencies):.1f} sn")

    if cfg.save_alerts and alerts:
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        out = ALERTS_DIR / f"stream_{strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(
            json.dumps({"source": cfg.source, "alerts": alerts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Kayıt: {out}")


if __name__ == "__main__":
    main()
