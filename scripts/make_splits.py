#!/usr/bin/env python3
"""Gold kümesini katmanlı biçimde dev / test olarak ayırır.

dev  = prompt, eşik ve kural ayarlarını burada yaparız
test = ayar yaparken dokunmayız, sunumdaki sayı buradan gelir

Katman: (category, ambiguity) çifti. Böylece "belirsiz kaza" videoları iki
kümeye de dengeli dağılır, test kümesi kolay videolardan oluşmaz.

    python scripts/make_splits.py                      # test payı 0.40
    python scripts/make_splits.py --test-ratio 0.5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.kpi import normalize_ambiguity  # noqa: E402
from utils.splits import SPLITS_PATH  # noqa: E402

GOLD_DEFAULT = ROOT / "data" / "exports" / "gold_labels_hepsi.json"


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--out", type=Path, default=SPLITS_PATH)
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.gold.read_text(encoding="utf-8"))
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        name = row.get("filename") or row.get("video_id")
        if not name:
            continue
        key = (row.get("category") or "?", normalize_ambiguity(row.get("ambiguity")))
        strata[key].append(str(name))

    rng = random.Random(args.seed)
    dev: list[str] = []
    test: list[str] = []
    for key in sorted(strata):
        names = sorted(strata[key])
        rng.shuffle(names)
        n_test = int(round(len(names) * args.test_ratio))
        # Katmanda tek video varsa dev'e gitsin; test kümesi tek örnekle şişmesin
        n_test = min(n_test, max(len(names) - 1, 0))
        test.extend(names[:n_test])
        dev.extend(names[n_test:])

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "gold": str(args.gold),
        "dev": sorted(dev),
        "test": sorted(test),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def dist(names: list[str]) -> str:
        by_name = {str(r.get("filename") or r.get("video_id")): r for r in rows}
        counts = Counter(
            (by_name[n].get("category"), normalize_ambiguity(by_name[n].get("ambiguity")))
            for n in names
            if n in by_name
        )
        return ", ".join(f"{cat}/{amb}={n}" for (cat, amb), n in sorted(counts.items()))

    print(f"dev  : {len(dev):3} video | {dist(dev)}")
    print(f"test : {len(test):3} video | {dist(test)}")
    print(f"Yazıldı: {args.out}")
    print("\nKullanım:")
    print("  python scripts/evaluate_kpi.py --pred data/predictions_wide --split dev")
    print("  python scripts/evaluate_kpi.py --pred data/predictions_wide --split test   # raporlanan sayı")


if __name__ == "__main__":
    main()
