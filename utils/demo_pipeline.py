"""Demo çekirdeği: bir video + jürinin promptu → zaman damgalı İSG cevabı.

İki kullanım:
    - Sunum sahnesi (4 dk + 1 dk): kendi videomuz, Hızlı mod açık olabilir.
    - Jüri videosu: süre sınırı söylenmedi; ikinci bakış / eleştirmen / RAG
      atlanmasın diye varsayılan bütçe uzun (time_budget_s). Hızlı mod kapalı.

Adımlar
    1. Kare çıkarma (hareket tepelerine göre) ve sensör kanıtı — eşzamanlı
    2. VLM: yapılandırılmış etiket (özet / olaylar / risk / aksiyon)
    3. Şüpheli sahnede kısa ikinci bakış
    4. Çelişkide karesiz metin eleştirmeni + kural katmanı
    5. LLM: jürinin sorusuna düz Türkçe cevap + saha aksiyonları (RAG referanslı)

CLI: scripts/analyze_video.py, arayüz: app/demo_app.py — ikisi de burayı çağırır.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import extract_video, safe_id  # noqa: E402

from utils import config  # noqa: E402
from utils.display import (  # noqa: E402
    attach_hard_case_sentence,
    hard_case_note,
    law_support_note,
    verdict,
)
from utils.label_json import (  # noqa: E402
    collapse_cloned_events,
    dedupe_events,
    label_to_spec,
    parse_json,
    preferred_incident_peak_s,
    snap_events_to_frame_times,
)
from utils.model_client import (  # noqa: E402
    CALL_LOG,
    ModelCallError,
    call_log_rows,
    chat_llm,
    chat_vlm,
    reset_call_log,
)
from agents.label_critic import critique_label, needs_critic  # noqa: E402
from utils.risk_rules import (  # noqa: E402
    needs_second_look,
    refine_label,
    scene_is_process_routine,
)
from utils.scene_evidence import SceneEvidence, analyze_video  # noqa: E402
from utils.spec_output import seconds_to_mmss  # noqa: E402

VIDEO_PROMPT_PATH = ROOT / "prompts" / "video_label_prompt.txt"
ANSWER_PROMPT_PATH = ROOT / "prompts" / "demo_answer_prompt.txt"
DEMO_RUNS_DIR = ROOT / "data" / "demo_runs"


def _run_result_path(run_dir: Path) -> Path | None:
    for name in ("result.json", "result.json", "result.json"):
        path = run_dir / name
        if path.is_file():
            return path
    extras = [p for p in run_dir.glob("*.json") if p.name != "spec.json"]
    return extras[0] if extras else None


def list_saved_runs(limit: int = 8) -> list[Path]:
    """En yeni demo yedek klasörleri (result.json / result.json olanlar)."""
    if not DEMO_RUNS_DIR.is_dir():
        return []
    runs = [
        path
        for path in DEMO_RUNS_DIR.iterdir()
        if path.is_dir() and _run_result_path(path) is not None
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]


def load_saved_run(run_dir: Path) -> DemoResult:
    payload = _run_result_path(run_dir)
    if payload is None:
        raise FileNotFoundError(f"Yedek JSON yok: {run_dir}")
    raw = json.loads(payload.read_text(encoding="utf-8"))
    result = DemoResult.from_dict(raw)
    result.out_dir = str(run_dir)
    return result


FRAMES_DIR = ROOT / "data" / "frames"

DEFAULT_PROMPT = "Bu videoda bir iş kazası var mı? Varsa ne olduğunu ve kaçıncı saniyede olduğunu söyle."

# Bundan uzun videolarda kareler tüm videoya yayılmaz; wake-up (hareket) tepesinin
# çevresine odaklanır. Hem süre hem olay yakalama doğruluğu için.
LONG_VIDEO_S: float = 60.0
WAKE_WINDOW_S: float = 10.0
# Tepeler birbirinden çok uzaksa pencere tüm videoya yayılmasın
MAX_WAKE_SPAN_S: float = 45.0

SECOND_LOOK_PROMPT = (
    "Önceki cevabın çok sakin / düşük risk görünüyor ama sensörler şüpheli diyor.\n"
    "Sadece bu karelere tekrar bak. Sırayla kontrol et:\n"
    "1) Yük / kalıp çalışanın ÜSTÜNE mi düştü, yoksa YANINA mı?\n"
    "2) Forklift kabloya, tavana veya iskeleye çarptı mı / bir şey çöktü mü?\n"
    "3) Çarpışma, düşme, yanma, kişi-araç temas var mı?\n"
    "Görmüyorsan uydurma. Görüyorsan category/risk'i yükselt.\n"
    "events[].time yalnızca verilen kare zamanlarından biri olsun.\n"
    "Yine sadece JSON döndür.\n"
)

SECOND_LOOK_PROMPT_VIDEO = (
    "Önceki cevabın çok sakin / düşük risk görünüyor ama sensörler şüpheli diyor.\n"
    "Aynı video klibine tekrar bak. Sırayla kontrol et:\n"
    "1) Yük / kalıp çalışanın ÜSTÜNE mi düştü, yoksa YANINA mı?\n"
    "2) Forklift kabloya, tavana veya iskeleye çarptı mı / bir şey çöktü mü?\n"
    "3) Çarpışma, düşme, yanma, kişi-araç temas var mı?\n"
    "Görmüyorsan uydurma. Görüyorsan category/risk'i yükselt.\n"
    "events[].time yalnızca verilen kare zamanlarından biri olsun.\n"
    "Yine sadece JSON döndür.\n"
)


@dataclass
class DemoResult:
    """Demo koşusunun tüm çıktısı; JSON'a bire bir yazılır."""

    video: str
    user_prompt: str
    answer: str
    spec: dict[str, Any]
    label: dict[str, Any]
    frames: list[dict[str, Any]]
    evidence: dict[str, Any]
    timings: dict[str, float]
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    total_s: float = 0.0
    fast_mode: bool = False
    out_dir: str = ""
    warnings: list[str] = field(default_factory=list)
    law_note: str = ""
    law_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "user_prompt": self.user_prompt,
            "answer": self.answer,
            "spec": self.spec,
            "label": self.label,
            "frames": self.frames,
            "evidence": self.evidence,
            "timings": self.timings,
            "model_calls": self.model_calls,
            "provider": self.provider,
            "total_s": round(self.total_s, 2),
            "fast_mode": self.fast_mode,
            "out_dir": self.out_dir,
            "warnings": self.warnings,
            "law_note": self.law_note,
            "law_detail": self.law_detail,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DemoResult:
        """data/demo_runs/.../result.json yedeğini ekrana geri yükler."""
        return cls(
            video=str(raw.get("video") or ""),
            user_prompt=str(raw.get("user_prompt") or ""),
            answer=str(raw.get("answer") or ""),
            spec=dict(raw.get("spec") or {}),
            label=dict(raw.get("label") or {}),
            frames=list(raw.get("frames") or []),
            evidence=dict(raw.get("evidence") or {}),
            timings=dict(raw.get("timings") or {}),
            model_calls=list(raw.get("model_calls") or []),
            provider=str(raw.get("provider") or ""),
            total_s=float(raw.get("total_s") or 0.0),
            fast_mode=bool(raw.get("fast_mode")),
            out_dir=str(raw.get("out_dir") or ""),
            warnings=list(raw.get("warnings") or []),
            law_note=str(raw.get("law_note") or ""),
            law_detail=str(raw.get("law_detail") or ""),
        )

    @classmethod
    def coerce(cls, raw: Any) -> DemoResult:
        """Streamlit eski oturum nesnesini / dict yedeği yeni alana çevirir."""
        if raw is None:
            raise TypeError("DemoResult yok")
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        return cls.from_dict(
            {
                "video": getattr(raw, "video", "") or "",
                "user_prompt": getattr(raw, "user_prompt", "") or "",
                "answer": getattr(raw, "answer", "") or "",
                "spec": dict(getattr(raw, "spec", None) or {}),
                "label": dict(getattr(raw, "label", None) or {}),
                "frames": list(getattr(raw, "frames", None) or []),
                "evidence": dict(getattr(raw, "evidence", None) or {}),
                "timings": dict(getattr(raw, "timings", None) or {}),
                "model_calls": list(getattr(raw, "model_calls", None) or []),
                "provider": getattr(raw, "provider", "") or "",
                "total_s": float(getattr(raw, "total_s", 0.0) or 0.0),
                "fast_mode": bool(getattr(raw, "fast_mode", False)),
                "out_dir": getattr(raw, "out_dir", "") or "",
                "warnings": list(getattr(raw, "warnings", None) or []),
                "law_note": getattr(raw, "law_note", "") or "",
                "law_detail": getattr(raw, "law_detail", "") or "",
            }
        )

    def report_text(self) -> str:
        """Jüriye okunacak düz metin rapor."""
        events = self.spec.get("events") or []
        lines = [
            "İSG VİDEO ANALİZ RAPORU",
            "=" * 56,
            f"Video      : {Path(self.video).name}",
            f"Soru       : {self.user_prompt or '(standart İSG taraması)'}",
            f"Süre       : {self.total_s:.1f} sn  (model: {self.provider})",
            "-" * 56,
            f"CEVAP      : {self.answer}",
            "-" * 56,
            f"Risk       : {self.spec.get('risk', '-')}  "
            f"(kategori: {self.label.get('category', '-')})",
            f"Özet       : {self.spec.get('summary', '-')}",
        ]
        note = hard_case_note(self.label, self.spec, self.evidence)
        if note:
            lines.append(f"Zor sahne  : {note['text']}")
        lines.append("Olaylar    :")
        if events:
            lines.extend(f"  {item.get('time', '00:00')}  {item.get('event', '')}" for item in events)
        else:
            lines.append("  (olay yok)")
        lines.append("Aksiyonlar :")
        actions = self.spec.get("actions") or []
        if actions:
            lines.extend(f"  {i}. {action}" for i, action in enumerate(actions, start=1))
        else:
            lines.append("  (aksiyon yok)")
        if self.law_note:
            lines.append(f"Mevzuat    : {self.law_note}")
        if self.law_detail:
            for line in self.law_detail.splitlines():
                if line.strip():
                    lines.append(f"             {line.strip()}")
        lines.append("-" * 56)
        lines.append("Aşama süreleri: " + ", ".join(f"{k}={v:.1f}s" for k, v in self.timings.items()))
        if self.warnings:
            lines.append("Uyarılar: " + " | ".join(self.warnings))
        return "\n".join(lines)


_EMERGENCY_ACT = ("sağlık", "itfaiye", "tahliye", "boşalt", "acil tıbbi")


def polish_demo_result(result: DemoResult) -> DemoResult:
    """Eski oturum + kural kaçaklarını ekranda düzelt: proses alevi, kopya zaman, mevzuat notu."""
    result = DemoResult.coerce(result)
    spec = dict(result.spec or {})
    label = dict(result.label or {})
    evidence = dict(result.evidence or {})
    peak_raw = evidence.get("motion_peak_sec")
    try:
        peak_s = float(peak_raw) if peak_raw is not None else None
    except (TypeError, ValueError):
        peak_s = None
    summary = str(spec.get("summary") or label.get("summary") or "")
    events = collapse_cloned_events(
        list(spec.get("events") or label.get("events") or []),
        peak_s=peak_s,
        summary=f"{summary} {result.answer}",
    )
    spec["events"] = events
    label["events"] = events
    if scene_is_process_routine(label, spec, result.answer):
        label["category"] = "normal"
        spec["risk"] = "Düşük"
        label["risk"] = "Düşük"
        acts = [str(a) for a in (spec.get("actions") or []) if a]
        if any(any(key in a.casefold() for key in _EMERGENCY_ACT) for a in acts):
            spec["actions"] = ["Rutin izlemeye devam et"]
            label["actions"] = spec["actions"]
    note = (result.law_note or "").strip()
    detail = (result.law_detail or "").strip()
    if not note or not detail:
        from utils.evren_rag import retrieve_mevzuat_lexical

        rag = retrieve_mevzuat_lexical(
            f"{result.answer} {summary} {spec.get('risk')} "
            f"{' '.join(str(a) for a in (spec.get('actions') or []))}"
        )
        if not note:
            note = law_support_note(rag)
        if not detail:
            detail = rag
    return DemoResult(
        video=result.video,
        user_prompt=result.user_prompt,
        answer=result.answer,
        spec=spec,
        label=label,
        frames=result.frames,
        evidence=result.evidence,
        timings=result.timings,
        model_calls=result.model_calls,
        provider=result.provider,
        total_s=result.total_s,
        fast_mode=result.fast_mode,
        out_dir=result.out_dir,
        warnings=result.warnings,
        law_note=note,
        law_detail=detail,
    )


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def use_utf8_stdout() -> None:
    """Windows konsolu Türkçe / özel karakterde demo ortasında patlamasın."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def _build_vlm_prompt(
    frames_meta: dict[str, Any],
    evidence: SceneEvidence,
    user_prompt: str,
    clip_span: tuple[float, float] | None = None,
) -> str:
    """Jürinin sorusu + kare zamanları + sensör kanıtı + şartname şeması."""
    frames = frames_meta["frames"]
    times = ", ".join(frame["time"] for frame in frames)
    numbered = "\n".join(
        f"{i}. kare zamanı {frame['time']} (events[].time için aday)"
        for i, frame in enumerate(frames, start=1)
    )
    question = user_prompt.strip()
    question_block = (
        "Operatörün / jürinin sorusu (etiketin bu soruyu karşılamalı):\n"
        f"\"{question}\"\n"
        "Soruyu cevaplarken de yalnızca gördüğünü yaz.\n\n"
        if question
        else ""
    )
    if clip_span:
        start_txt, end_txt = seconds_to_mmss(clip_span[0]), seconds_to_mmss(clip_span[1])
        media_line = (
            f"Gönderilen medya: orijinal videonun {start_txt}-{end_txt} aralığından "
            "kesilmiş kısa video klibi (kare dizisi değil). Tüm klibi izle. "
            "events[].time orijinal video saatine göre yaz (yukarıdaki kare zamanları); "
            "klibin ilk anına 00:00 deme. "
        )
    else:
        media_line = "Görseller aşağıda 1. kareden son kareye sıralı. "
    return (
        f"{question_block}"
        f"Süre: {frames_meta['duration_sec']} saniye\n"
        f"Kare zamanları (events[].time SADECE bunlardan biri): {times}\n"
        f"Gönderilen kare sayısı: {len(frames)}\n"
        f"{evidence.prompt_block()}\n"
        f"{media_line}"
        "İlk an sakin olsa bile tüm diziyi oku; "
        "çarpışma, düşme, yanma, devrilme veya yerde kişi varsa onu yaz "
        "(accident + Yüksek). Sadece tehlikeli yaklaşma varsa near_miss.\n"
        f"{numbered}\n\n" + _load(VIDEO_PROMPT_PATH)
    )


def _pick_focus_frames(frames: list[dict[str, Any]], evidence: SceneEvidence) -> list[dict[str, Any]]:
    """İkinci bakış için hareket tepelerine yakın en fazla 3 kare.

    Tek tepeye bakmak yetmiyor: ölçümde en yüksek tepe gold olay anını ±2 sn içinde
    sadece %25 buluyor, ilk üç tepeden biri %69 buluyor (scripts/eval_wakeup.py).
    """
    if len(frames) <= 3:
        return frames
    peaks = evidence.motion_peaks or (
        [evidence.motion_peak_sec] if evidence.motion_peak_sec is not None else []
    )
    if not peaks:
        return frames[-3:]
    ranked = sorted(
        frames,
        key=lambda f: min(abs(float(f.get("t_sec", 0.0)) - peak) for peak in peaks),
    )
    return sorted(ranked[:3], key=lambda f: float(f.get("t_sec", 0.0)))


def _label_from_parsed(
    parsed: dict[str, Any],
    frames_meta: dict[str, Any],
    evidence: SceneEvidence,
    model_label: str,
) -> dict[str, Any]:
    frame_times = [frame["time"] for frame in frames_meta["frames"]]
    category = parsed.get("category")
    if category not in {"normal", "near_miss", "accident"}:
        category = "normal"
    return {
        "video_id": frames_meta["video_id"],
        "filename": Path(frames_meta["video"]).name,
        "category": category,
        "duration_sec": frames_meta["duration_sec"],
        "status": "demo",
        "labeled_by": model_label,
        "summary": parsed.get("summary", ""),
        "events": dedupe_events(
            snap_events_to_frame_times(parsed.get("events") or [], frame_times)
        ),
        "risk": parsed.get("risk", "Orta"),
        "actions": [a for a in (parsed.get("actions") or []) if a],
        "notes": "Demo koşusu (canlı gösterim çıktısı).",
        "evidence": evidence.to_dict(),
    }


def _fallback_answer(
    spec: dict[str, Any],
    label: dict[str, Any],
    evidence: SceneEvidence | dict[str, Any] | None = None,
) -> str:
    """LLM cevap adımı düşerse yapılandırılmış bulgudan cümle kurar."""
    events = spec.get("events") or []
    v = verdict(label.get("category"), spec.get("risk") or label.get("risk"))
    first = events[0] if events else None
    when = f" İlk kritik an {first['time']}." if first else ""
    what = f" {first.get('event')}" if first and first.get("event") else ""
    raw = (
        f"Videoda {v['situation'].lower()} görüldü. "
        f"Karar: {v['decision']}.{when}{what}"
    ).strip()
    ev = evidence.to_dict() if isinstance(evidence, SceneEvidence) else evidence
    return attach_hard_case_sentence(raw, hard_case_note(label, spec, ev))


def probe_duration(video_path: Path) -> float:
    """Kare çıkarmadan önce süreyi öğrenir (uzun video kararı için, ~ms sürer)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frame_count / fps) if fps else 0.0


def wake_window(evidence: SceneEvidence, duration: float) -> tuple[float, float] | None:
    """Uzun videoda aday pencere: ilk üç hareket tepesini kapsayan aralık.

    Tek tepeye ±10 sn açmak riskliydi; ölçümde olayların %21'i o pencerenin dışında
    kalıyordu. Tepe listesini kapsayan aralık daha güvenli, pencere içindeki kare
    seçimi zaten hareket skoruna göre yapıldığı için genişlik maliyet değil.
    """
    if duration <= LONG_VIDEO_S:
        return None
    peaks = [p for p in (evidence.motion_peaks or []) if p is not None]
    if not peaks and evidence.motion_peak_sec is not None:
        peaks = [evidence.motion_peak_sec]
    if not peaks:
        return (max(0.0, duration / 2), duration)
    start = max(0.0, min(peaks) - WAKE_WINDOW_S)
    end = min(duration, max(peaks) + WAKE_WINDOW_S)
    if end - start > MAX_WAKE_SPAN_S:
        center = sum(peaks) / len(peaks)
        start = max(0.0, center - MAX_WAKE_SPAN_S / 2)
        end = min(duration, start + MAX_WAKE_SPAN_S)
    return (start, end)


async def _rag_context(query: str) -> str:
    """Mevzuat referansı; EVREN embed yoksa Chroma, o da yoksa boş."""
    try:
        from utils.evren_rag import retrieve_mevzuat

        text = await retrieve_mevzuat(query)
        if text:
            return text
    except Exception as exc:
        print(f"  [rag] EVREN atlandı: {exc}")
    try:
        from agents.action_recommender import retrieve_isg_context

        return await asyncio.to_thread(retrieve_isg_context, query)
    except Exception as exc:
        print(f"  [rag] atlandı: {exc}")
        return ""


async def _answer_step(
    spec: dict[str, Any],
    label: dict[str, Any],
    evidence: SceneEvidence,
    user_prompt: str,
) -> tuple[str, list[str], str]:
    """(cevap, aksiyonlar, uyarı) döner."""
    question = user_prompt.strip() or DEFAULT_PROMPT
    note = hard_case_note(label, spec, evidence.to_dict())
    findings = {
        "kategori": label.get("category"),
        "risk": spec.get("risk"),
        "ozet": spec.get("summary"),
        "olaylar": spec.get("events"),
        "model_aksiyonlari": spec.get("actions"),
        "sensor": {
            "hareket_tepesi": seconds_to_mmss(evidence.motion_peak_sec)
            if evidence.motion_peak_sec is not None
            else None,
            "kisi_arac_cok_yakin": evidence.person_vehicle_very_close,
            "yangin_suphesi": evidence.fire_suspect,
        },
    }
    if note:
        findings["zor_sahne"] = note["text"]
    prompt = (
        f"Soru: {question}\n\n"
        f"Bulgular (JSON): {json.dumps(findings, ensure_ascii=False)}\n"
        "Birden fazla olay varsa cevapta SON kaza/düşme zamanını kullan "
        "(erken sallanmayı düşme sanma).\n"
    )

    try:
        result = await chat_llm(
            prompt,
            system=_load(ANSWER_PROMPT_PATH),
            temperature=0.1,
            max_tokens=320,
            json_mode=True,
        )
        parsed = parse_json(result.text)
        answer = str(parsed.get("answer") or "").strip()
        actions = [str(a) for a in (parsed.get("actions") or []) if str(a).strip()]
        if not answer:
            raise ValueError("boş answer")
        return attach_hard_case_sentence(answer, note), actions, ""
    except (ModelCallError, ValueError) as exc:
        warning = f"Cevap adımı LLM'siz üretildi: {exc}"
        print(f"  [cevap] {warning}")
        return _fallback_answer(spec, label, evidence), spec.get("actions") or [], warning


async def run_demo_analysis(
    video: Path | str,
    user_prompt: str = "",
    *,
    max_frames: int | None = None,
    every_sec: float = 0.5,
    fast: bool | None = None,
    use_rag: bool = True,
    use_second_look: bool = True,
    time_budget_s: float = 600.0,
    save: bool = True,
    out_root: Path | None = None,
    run_name: str = "",
    progress: Callable[[str], None] | None = None,
) -> DemoResult:
    """Tek videoyu uçtan uca analiz eder ve DemoResult döner."""
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    fast_mode = config.demo_fast_mode() if fast is None else fast
    frames_wanted = max_frames or config.demo_max_frames()
    warnings: list[str] = []
    timings: dict[str, float] = {}
    reset_call_log()

    def say(message: str) -> None:
        print(message, flush=True)
        if progress:
            progress(message)

    started = perf_counter()
    say(f"[1/5] Kare çıkarma + sensör kanıtı ({frames_wanted} kare hedefi)")
    step = perf_counter()
    duration = await asyncio.to_thread(probe_duration, video_path)

    if duration > LONG_VIDEO_S:
        # Uzun videoda önce kanıt: hareket tepesi kareleri hangi aralıktan alacağımızı belirler
        evidence = await asyncio.to_thread(analyze_video, video_path, 24, use_yolo=not fast_mode)
        window = wake_window(evidence, duration)
        if window:
            say(
                f"      uzun video ({duration:.0f}s), wake-up penceresi: "
                f"{seconds_to_mmss(window[0])}-{seconds_to_mmss(window[1])}"
            )
        frames_meta = await asyncio.to_thread(
            extract_video, video_path, FRAMES_DIR, every_sec, frames_wanted, window=window
        )
    else:
        frames_meta, evidence = await asyncio.gather(
            asyncio.to_thread(
                extract_video, video_path, FRAMES_DIR, every_sec, frames_wanted, use_motion=True
            ),
            asyncio.to_thread(analyze_video, video_path, 24, use_yolo=not fast_mode),
        )

    timings["kare_ve_kanit"] = perf_counter() - step
    frame_times = [frame["time"] for frame in frames_meta["frames"]]
    say(f"      {len(frame_times)} kare: {', '.join(frame_times)}")

    say("[2/5] VLM analizi")
    step = perf_counter()
    frame_paths = [Path(frame["path"]) for frame in frames_meta["frames"]]
    clip_path: Path | None = None
    clip_range: tuple[float, float] | None = None
    if config.provider() == "teknofest":
        from utils.video_clip import prepare_clip

        dest = ROOT / "data" / "clips" / f"{safe_id(video_path)}.mp4"
        duration_s = float(frames_meta.get("duration_sec") or duration)
        clip_path, clip_start, clip_end = await asyncio.to_thread(
            prepare_clip,
            video_path,
            dest,
            duration_s,
            evidence.motion_peak_sec,
            evidence.motion_peaks,
        )
        clip_range = (clip_start, clip_end)
        say(
            f"      EVREN klibi {seconds_to_mmss(clip_start)}-{seconds_to_mmss(clip_end)} "
            f"({clip_path.stat().st_size / 1e6:.1f} MB)"
        )
    vlm = await chat_vlm(
        _build_vlm_prompt(frames_meta, evidence, user_prompt, clip_range),
        frame_paths,
        video_path=clip_path,
        temperature=0.1,
        max_tokens=768,
        json_mode=True,
    )
    timings["vlm"] = perf_counter() - step
    try:
        parsed = parse_json(vlm.text)
    except ValueError as exc:
        warnings.append(f"VLM JSON ayrıştırılamadı, düz metne düşüldü: {exc}")
        parsed = {"summary": " ".join(vlm.text.split())[:300], "events": [], "risk": "Orta"}
    label = _label_from_parsed(parsed, frames_meta, evidence, f"{vlm.provider}:{vlm.model}")
    say(f"      risk={label.get('risk')} kategori={label.get('category')} ({timings['vlm']:.1f}s)")

    elapsed = perf_counter() - started
    second_look_ok = (
        use_second_look
        and not fast_mode
        and elapsed < time_budget_s * 0.5
        and needs_second_look(label, evidence)
    )
    if second_look_ok:
        say("[3/5] İkinci bakış (sensör şüpheli, model sakin)")
        step = perf_counter()
        focus = _pick_focus_frames(frames_meta["frames"], evidence)
        try:
            if clip_path:
                second_prompt = (
                    f"{SECOND_LOOK_PROMPT_VIDEO}\n{evidence.prompt_block()}\n"
                    f"Odak zamanları: {', '.join(f['time'] for f in focus)}\n"
                    f"Önceki JSON özeti: risk={label.get('risk')}, summary={label.get('summary')}\n\n"
                    + _load(VIDEO_PROMPT_PATH)
                )
                second = await chat_vlm(
                    second_prompt,
                    frame_paths,
                    video_path=clip_path,
                    temperature=0.1,
                    max_tokens=640,
                    json_mode=True,
                )
            else:
                second = await chat_vlm(
                    f"{SECOND_LOOK_PROMPT}\n{evidence.prompt_block()}\n"
                    f"Odak kareler: {', '.join(f['time'] for f in focus)}\n"
                    f"Önceki JSON özeti: risk={label.get('risk')}, summary={label.get('summary')}\n\n"
                    + _load(VIDEO_PROMPT_PATH),
                    [Path(frame["path"]) for frame in focus],
                    temperature=0.1,
                    max_tokens=640,
                    json_mode=True,
                )
            parsed2 = parse_json(second.text)
            for key in ("summary", "events", "risk", "actions", "category"):
                if parsed2.get(key) not in (None, "", []):
                    label[key] = parsed2[key]
            label["events"] = dedupe_events(
                snap_events_to_frame_times(
                    label.get("events") or [], [frame["time"] for frame in focus] or frame_times
                )
            )
            label["notes"] = f"{label.get('notes', '')} | ikinci bakış uygulandı".strip(" |")
        except (ModelCallError, ValueError) as exc:
            warnings.append(f"İkinci bakış atlandı: {exc}")
        timings["ikinci_bakis"] = perf_counter() - step
    else:
        say("[3/5] İkinci bakış gerekmedi (hızlı mod veya kanıt sakin)")

    elapsed = perf_counter() - started
    critic_ok = (
        config.label_critic_llm()
        and not fast_mode
        and elapsed < time_budget_s * 0.7
        and needs_critic(label)
    )
    if critic_ok:
        say("      metin eleştirmeni (kare yok, yalnız çelişki)")
        step = perf_counter()
        try:
            label = await critique_label(label)
        except ModelCallError as exc:
            warnings.append(f"Eleştirmen atlandı: {exc}")
        timings["elestirmen"] = perf_counter() - step

    say("[4/5] Kural katmanı (zaman hizalama + kanıt birleştirme)")
    step = perf_counter()
    label = refine_label(label, evidence)
    peak_s = preferred_incident_peak_s(
        list(evidence.motion_peaks or []),
        evidence.motion_peak_sec,
        duration_s=evidence.duration_sec,
        category=str(label.get("category") or ""),
    )
    label["events"] = collapse_cloned_events(
        dedupe_events(
            snap_events_to_frame_times(label.get("events") or [], frame_times),
            window_sec=2,
            max_events=5,
        ),
        peak_s=peak_s,
        summary=str(label.get("summary") or ""),
    )
    spec = label_to_spec(label)
    timings["kural_katmani"] = perf_counter() - step

    say("[5/5] Cevap + saha aksiyonları (LLM)")
    step = perf_counter()
    rag_text = ""
    rag_ok = use_rag and not fast_mode and (perf_counter() - started) < time_budget_s * 0.8
    if rag_ok:
        rag_text = await _rag_context(f"{spec.get('summary')} Risk: {spec.get('risk')}")
    answer, actions, warning = await _answer_step(spec, label, evidence, user_prompt)
    law_note = law_support_note(rag_text)
    if warning:
        warnings.append(warning)
    if actions:
        label["actions"] = actions
        spec["actions"] = actions
    timings["cevap"] = perf_counter() - step

    total = perf_counter() - started
    if total > 60.0:
        warnings.append(f"Süre 60 sn'yi aştı ({total:.0f}s). DEMO_FAST_MODE=1 ile tekrar deneyin.")

    result = DemoResult(
        video=str(video_path),
        user_prompt=user_prompt,
        answer=answer,
        spec=spec,
        label=label,
        frames=frames_meta["frames"],
        evidence=evidence.to_dict(),
        timings={key: round(value, 2) for key, value in timings.items()},
        model_calls=call_log_rows(),
        provider=CALL_LOG[0].provider if CALL_LOG else config.provider(),
        total_s=total,
        fast_mode=fast_mode,
        warnings=warnings,
        law_note=law_note,
        law_detail=rag_text,
    )
    result = polish_demo_result(result)

    if save:
        out_dir = _write_run(result, out_root or DEMO_RUNS_DIR, run_name or safe_id(video_path))
        result.out_dir = str(out_dir)
        say(f"Kaydedildi: {out_dir}")

    v = verdict(result.label.get("category"), result.spec.get("risk"))
    say(
        f"Toplam süre: {total:.1f} sn | {v['situation']} · {v['decision']} "
        f"(şartname: {v['spec_risk']}) | Cevap: {answer}"
    )
    return result


def _write_run(result: DemoResult, out_root: Path, run_name: str) -> Path:
    """result.json + report.txt + kanıt kareleri (self-contained klasör)."""
    out_dir = out_root / run_name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame in result.frames:
        src = Path(frame["path"])
        if src.exists():
            dest = frames_dir / f"{frame['time'].replace(':', '-')}{src.suffix}"
            shutil.copy2(src, dest)
            frame["demo_path"] = str(dest)

    (out_dir / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "spec.json").write_text(
        json.dumps(result.spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.txt").write_text(result.report_text(), encoding="utf-8")
    return out_dir


def run_demo_analysis_sync(video: Path | str, user_prompt: str = "", **kwargs: Any) -> DemoResult:
    """Streamlit / senkron çağrılar için sarmalayıcı."""
    return asyncio.run(run_demo_analysis(video, user_prompt, **kwargs))
