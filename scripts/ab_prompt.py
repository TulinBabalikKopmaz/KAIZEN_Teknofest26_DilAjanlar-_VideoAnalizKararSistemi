#!/usr/bin/env python3
"""Prompt A/B ölçümü: iki prompt dosyasını aynı videolarda karşılaştırır.

Neden: terminoloji sözlüğü gerçekten metin eşleşmesini açıyor mu, yoksa biz mi
öyle sanıyoruz? Aynı model, aynı videolar, tek değişken prompt.

    python scripts/ab_prompt.py --backend ollama --n 18
    python scripts/ab_prompt.py --backend teknofest --n all

Her kol ayrı klasöre yazar (data/predictions_ab_a, _b) ve sonunda iki KPI raporu
yan yana basılır. Kol başına süre ~ video sayısı x model gecikmesi.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402

METRICS = (
    ("risk_accuracy", "Risk doğruluğu"),
    ("event_recall", "Olay yakalama"),
    ("critical_recall", "Kritik olay"),
    ("false_alarm_rate", "Yanlış alarm"),
    ("summary_ok_rate", "Özet benzerliği"),
    ("action_ok_rate", "Aksiyon doluluğu"),
)


def run_arm(name: str, prompt_file: Path, args: argparse.Namespace) -> Path:
    pred_dir = ROOT / "data" / f"predictions_ab_{name}"
    env = {**os.environ, "VIDEO_PROMPT_FILE": str(prompt_file)}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_kpi_wide.py"),
        "--backend",
        args.backend,
        "--n",
        str(args.n),
        "--tag",
        f"ab_{name}",
        "--pred-dir",
        str(pred_dir),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print("\n" + "=" * 70)
    print(f"KOL {name.upper()}  prompt={prompt_file.name}")
    print("=" * 70)
    subprocess.call(cmd, env=env, cwd=str(ROOT))
    return pred_dir


def find_report(name: str) -> Path | None:
    matches = sorted((ROOT / "data" / "exports").glob(f"kpi_wide_*ab_{name}_report.json"))
    return matches[-1] if matches else None


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-a", type=Path, default=ROOT / "prompts" / "video_label_prompt_v1.txt")
    parser.add_argument("--prompt-b", type=Path, default=ROOT / "prompts" / "video_label_prompt.txt")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "teknofest", "openai"])
    parser.add_argument("--n", default="18", help="18 gibi sayı ya da 'all'")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--only-compare", action="store_true", help="Koşmadan mevcut raporları karşılaştır")
    args = parser.parse_args()

    for path in (args.prompt_a, args.prompt_b):
        if not path.exists():
            raise SystemExit(f"Prompt dosyası yok: {path}")

    if not args.only_compare:
        run_arm("a", args.prompt_a, args)
        run_arm("b", args.prompt_b, args)

    reports = {}
    for name in ("a", "b"):
        path = find_report(name)
        if path is None:
            print(f"Uyarı: {name} kolunun raporu bulunamadı.")
            continue
        reports[name] = json.loads(path.read_text(encoding="utf-8"))

    if len(reports) < 2:
        raise SystemExit("Karşılaştırma için iki rapor gerekli.")

    a = reports["a"]["summary"]
    b = reports["b"]["summary"]
    print("\n" + "=" * 70)
    print(f"A/B SONUÇ   A={args.prompt_a.name}   B={args.prompt_b.name}")
    print("=" * 70)
    print(f"  {'metrik':22} {'A':>7} {'B':>7} {'fark':>7}")
    for key, label in METRICS:
        va, vb = a.get(key, 0.0), b.get(key, 0.0)
        print(f"  {label:22} {va:>7.0%} {vb:>7.0%} {vb - va:>+7.0%}")
    print(f"\n  Video sayısı: A={a.get('n_video')}  B={b.get('n_video')}")
    print("  Yanlış alarm ve tahmini olmayan sayısı düşükse iyidir; diğerleri yüksekse iyi.")
    print("\n  Ayrıntı için: python scripts/diagnose_kpi.py --pred data/predictions_ab_b")


if __name__ == "__main__":
    main()
