"""dev / test ayrımı: ayarı dev üzerinde yap, sonucu test üzerinden raporla.

Aynı küme üzerinde prompt ve eşik ayarı yapıp sonra o kümede "başarı şu" demek,
kendi sınavının sorularını görmek demek. `data/exports/splits.json` bu ayrımı
kalıcı ve tekrar üretilebilir tutar (sabit tohum + katmanlı bölünme).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "exports" / "splits.json"


def load_splits(path: Path | None = None) -> dict[str, str]:
    """{video adı (uzantısız): 'dev' | 'test'}; dosya yoksa boş sözlük."""
    src = path or SPLITS_PATH
    if not src.exists():
        return {}
    data = json.loads(src.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split in ("dev", "test"):
        for name in data.get(split) or []:
            mapping[_key(name)] = split
    return mapping


def _key(name: str) -> str:
    stem = str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem


def split_of(row: dict[str, Any], mapping: dict[str, str]) -> str:
    for field in ("filename", "video_id"):
        value = row.get(field)
        if value and _key(str(value)) in mapping:
            return mapping[_key(str(value))]
    return "-"


def filter_by_split(
    rows: Iterable[dict[str, Any]],
    split: str,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """split 'dev' | 'test' | 'hepsi'. Ayrım dosyası yoksa hiçbir şeyi eleme."""
    items = list(rows)
    if split in {"", "hepsi", "all"}:
        return items
    mapping = load_splits(path)
    if not mapping:
        print("Uyarı: splits.json yok, --split yok sayıldı (scripts/make_splits.py çalıştırın).")
        return items
    return [row for row in items if split_of(row, mapping) == split]
