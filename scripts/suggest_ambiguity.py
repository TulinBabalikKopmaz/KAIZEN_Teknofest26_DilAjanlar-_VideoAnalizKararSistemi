#!/usr/bin/env python3
"""Belirsiz (zor okunan) videoları önerir; kararı insan verir.

Şartname test kümesinin ~%20'sinin belirsiz olmasını istiyor. Elimizdeki videolar
sonuçlarına göre doğru etiketlenmiş ama "bu sahne okunabilir mi" sorusu hiç
sorulmamış. 76 videoyu baştan izlemek yerine adayları sıralıyoruz; onay
`app/review_app.py` içinde tek tıkla veriliyor.

Sinyaller (hiçbiri tek başına kanıt değil, puan toplarlar):
    - normal video ama sensör alarm veriyor  → alarm görünümlü normal
    - kaza/ramak kala ama sahne sakin        → görüş engeli / zor görünür
    - model kategorisi gold ile çelişiyor    → gerçekten karıştırıcı sahne
    - gold notunda tereddüt ifadesi          → etiketçi zaten emin değilmiş
    - kısa video / düşük çözünürlük          → kalite

    python scripts/suggest_ambiguity.py                 # sadece rapor + CSV
    python scripts/suggest_ambiguity.py --apply --top 16  # taslak olarak işaretle (insan onayı bekler)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import safe_id  # noqa: E402

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.scene_evidence import analyze_video  # noqa: E402

GOLD_DEFAULT = ROOT / "data" / "exports" / "gold_labels_hepsi.json"
VIDEO_ROOT = ROOT / "data" / "videos"
LABEL_ROOT = ROOT / "data" / "labels"
PRED_DIRS = (ROOT / "data" / "predictions_wide", ROOT / "data" / "predictions")
OUT_CSV = ROOT / "data" / "exports" / "belirsizlik_onerileri.csv"

HESITATION = (
    "belli değil",
    "belli degil",
    "emin değil",
    "emin degil",
    "bulanık",
    "bulanik",
    "net değil",
    "net degil",
    "anlaşılmıyor",
    "anlasilmiyor",
    "görünmüyor",
    "gorunmuyor",
    "olabilir",
    "sanki",
    "muhtemelen",
    "şüpheli",
    "supheli",
)


def find_video(name: str) -> Path | None:
    for path in VIDEO_ROOT.rglob("*"):
        if path.name == name:
            return path
    return None


def load_pred_category(video_id: str, filename: str) -> str | None:
    for folder in PRED_DIRS:
        for candidate in (folder / f"{video_id}.json", folder / f"{Path(filename).stem}.json"):
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                return data.get("category")
    return None


def score_row(row: dict[str, Any], *, use_evidence: bool) -> dict[str, Any]:
    filename = str(row.get("filename") or "")
    video_id = str(row.get("video_id") or "")
    category = row.get("category") or "?"
    notes = f"{row.get('notes') or ''} {row.get('summary') or ''}".lower()

    score = 0.0
    reasons: list[str] = []
    proposed = ""

    if any(word in notes for word in HESITATION):
        score += 2.0
        reasons.append("gold notunda tereddüt")
        proposed = proposed or "niyet_belirsiz"

    pred_category = load_pred_category(video_id, filename)
    if pred_category and pred_category != category:
        score += 2.0
        reasons.append(f"model '{pred_category}' dedi, gold '{category}'")
        proposed = proposed or ("alarm_gorunumlu_normal" if category == "normal" else "gorus_engeli")

    video = find_video(filename)
    if video is None:
        reasons.append("video dosyası bulunamadı")
        return {
            "filename": filename,
            "category": category,
            "score": round(score, 2),
            "proposed_reason": proposed,
            "signals": "; ".join(reasons),
        }

    if use_evidence:
        evidence = analyze_video(video, 16, use_yolo=False)
        alarming = evidence.motion_elevated or evidence.fire_suspect
        if category == "normal" and alarming:
            score += 2.5
            reasons.append(
                "normal ama sensör alarm veriyor"
                + (" (yangın benzeri)" if evidence.fire_suspect else " (ani hareket)")
            )
            proposed = "alarm_gorunumlu_normal"
        if category in {"accident", "near_miss"} and not evidence.motion_elevated:
            score += 2.0
            reasons.append("olay var ama sahne sakin görünüyor")
            proposed = proposed or "gorus_engeli"

    import cv2

    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = frames / fps if fps else 0.0
        if width and width < 480:
            score += 1.0
            reasons.append(f"düşük çözünürlük ({int(width)}px)")
            proposed = proposed or "kalite"
        if 0 < duration < 4:
            score += 1.0
            reasons.append(f"çok kısa ({duration:.1f} sn)")
            proposed = proposed or "sonuc_gorunmuyor"
        if 0 < fps < 12:
            score += 0.5
            reasons.append(f"düşük fps ({fps:.0f})")
            proposed = proposed or "kalite"
    cap.release()

    return {
        "filename": filename,
        "category": category,
        "score": round(score, 2),
        "proposed_reason": proposed,
        "signals": "; ".join(reasons) or "-",
    }


def apply_suggestions(rows: list[dict[str, Any]], top: int) -> int:
    """Taslak olarak yazar: ambiguity_source='auto' → arayüzde onay bekler."""
    applied = 0
    for row in rows[:top]:
        if row["score"] <= 0:
            continue
        video = find_video(row["filename"])
        if video is None:
            continue
        path = LABEL_ROOT / f"{safe_id(video)}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("ambiguity_source") == "human":
            continue  # insan kararını ezmeyelim
        data["ambiguity"] = "belirsiz"
        data["ambiguity_reason"] = row["proposed_reason"] or "diger"
        data["ambiguity_source"] = "auto"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        applied += 1
    return applied


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--top", type=int, default=16, help="Kaç video işaretlenecek (hedef: ~yüzde 20)")
    parser.add_argument("--apply", action="store_true", help="Taslak olarak etiket dosyalarına yaz")
    parser.add_argument("--no-evidence", action="store_true", help="Sensör taramasını atla (hızlı)")
    args = parser.parse_args()

    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    print(f"{len(gold_rows)} gold video taranıyor...")
    scored = [score_row(row, use_evidence=not args.no_evidence) for row in gold_rows]
    scored.sort(key=lambda item: -item["score"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "category", "score", "proposed_reason", "signals"]
        )
        writer.writeheader()
        writer.writerows(scored)

    print(f"\nEn güçlü {min(args.top, len(scored))} aday:")
    for row in scored[: args.top]:
        if row["score"] <= 0:
            break
        print(f"  {row['score']:>4}  {row['category']:10} {row['filename'][:44]:44} {row['signals'][:60]}")

    candidates = sum(1 for row in scored if row["score"] > 0)
    print(f"\nSinyal veren video: {candidates} / {len(scored)}  (hedef belirsiz oranı ~%20)")
    print(f"Tablo: {args.out_csv}")

    if args.apply:
        applied = apply_suggestions(scored, args.top)
        print(f"\n{applied} videoya taslak belirsizlik işareti yazıldı (ambiguity_source=auto).")
        print("Onay için: streamlit run app/review_app.py → 'Sadece belirsizlik onayı bekleyenler'")
    else:
        print("\nİşaretlemek için: --apply ekleyin. Karar yine insanda; arayüzde onaylanır.")


if __name__ == "__main__":
    main()
