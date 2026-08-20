"""Sınıf bazlı accuracy / precision / recall / F1 (sklearn yok)."""

from __future__ import annotations

from typing import Any, Sequence


def class_report(
    pairs: Sequence[tuple[str | None, str | None]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """pairs: (gold, pred). pred None ise satır atlanır (eksik tahmin)."""
    rows = [(g, p) for g, p in pairs if g is not None and p is not None]
    n = len(rows)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    other = 0
    correct = 0
    for gold, pred in rows:
        if gold not in matrix or pred not in matrix[gold]:
            other += 1
            continue
        matrix[gold][pred] += 1
        if gold == pred:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other_label][label] for other_label in labels if other_label != label)
        fn = sum(matrix[label][other_label] for other_label in labels if other_label != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        support = sum(matrix[label].values())
        per_class[label] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    f1s = [per_class[label]["f1"] for label in labels if per_class[label]["support"]]
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "per_class": per_class,
        "confusion": matrix,
        "unmapped": other,
    }


def format_report(title: str, report: dict[str, Any], labels: Sequence[str]) -> str:
    lines = [
        f"{title}",
        f"  n={report['n']}  accuracy={report['accuracy']:.0%}  macro-F1={report['macro_f1']:.2f}",
        f"  {'sınıf':12} {'P':>6} {'R':>6} {'F1':>6} {'n':>4}",
    ]
    for label in labels:
        row = report["per_class"][label]
        lines.append(
            f"  {label:12} {row['precision']:6.0%} {row['recall']:6.0%} "
            f"{row['f1']:6.0%} {row['support']:4}"
        )
    header = "gold\\pred " + " ".join(f"{label:>8}" for label in labels)
    lines.append("  Karışıklık matrisi (satır=gold, sütun=tahmin):")
    lines.append("  " + header)
    for gold in labels:
        cells = " ".join(f"{report['confusion'][gold][pred]:8}" for pred in labels)
        lines.append(f"  {gold:10} {cells}")
    return "\n".join(lines)
