#!/usr/bin/env python3
"""Qwen-VL ile şartname JSON'una uygun taslak etiket üretir.

Çıktı gold değil, taslaktır. Mevcut gold dosyaların üzerine yazmaz.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import extract_video, iter_videos, safe_id
from utils.label_json import (
    dedupe_events,
    label_to_spec,
    parse_json,
    snap_events_to_frame_times,
)
from utils.risk_rules import needs_second_look, refine_label
from utils.scene_evidence import SceneEvidence, analyze_video

load_dotenv()

PROMPT_PATH = ROOT / "prompts" / "video_label_prompt.txt"
SECOND_LOOK_PROMPT = (
    "Önceki cevabın çok sakin / düşük risk görünüyor ama sensörler şüpheli diyor.\n"
    "Sadece bu karelere tekrar bak. Özellikle: çarpışma, düşme, yanma, kişi-araç temas.\n"
    "Görmüyorsan uydurma. Görüyorsan category/risk'i yükselt.\n"
    "events[].time yalnızca verilen kare zamanlarından biri olsun.\n"
    "Yine sadece JSON döndür.\n"
)


def load_prompt() -> str:
    """VIDEO_PROMPT_FILE ile başka bir prompt dosyası kullanılabilir (A/B ölçümü)."""
    override = os.getenv("VIDEO_PROMPT_FILE", "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path.read_text(encoding="utf-8")
        print(f"Uyarı: VIDEO_PROMPT_FILE bulunamadı ({path}), varsayılan prompt kullanılıyor.")
    return PROMPT_PATH.read_text(encoding="utf-8")


def encode_image(path: Path, max_side: int | None = None) -> str:
    data = path.read_bytes()
    if max_side:
        import cv2
        import numpy as np

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                data = buf.tobytes()
    return base64.b64encode(data).decode("ascii")


def ollama_base() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/").removesuffix("/v1")


def ollama_chat(model: str, prompt: str, image_paths: list[Path]) -> str:
    """Native Ollama /api/chat — OpenAI uyumlu katman num_ctx'i düşürüyor."""
    import time
    import urllib.error
    import urllib.request

    is_llava = "llava" in model.lower()
    default_ctx = "4096" if is_llava else "16384"
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", default_ctx))
    img_side = 320 if is_llava else None
    paths = list(image_paths)
    if is_llava and len(paths) > 2:
        paths = [paths[0], paths[-1]]
    text = prompt
    if is_llava and len(text) > 1200:
        text = text[:1200] + "\nSadece JSON döndür."

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": num_ctx,
            "num_predict": 512 if is_llava else 768,
            "temperature": 0.1,
        },
        "messages": [
            {
                "role": "user",
                "content": text,
                "images": [encode_image(path, max_side=img_side) for path in paths],
            }
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{ollama_base()}/api/chat"
    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=420) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content") or ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"Ollama HTTP {exc.code}: {detail[:500]}")
            retryable = exc.code in {500, 502, 503}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_err = RuntimeError(f"Ollama bağlantı hatası: {exc}")
            retryable = True
        if not retryable or attempt >= 3:
            break
        wait_s = 4 * attempt
        print(
            f"  Ollama hata (deneme {attempt}/3), {wait_s}s sonra tekrar…",
            flush=True,
        )
        # T4'te runner düşmüş olabilir — serve'i nazikçe yenile
        try:
            import subprocess

            subprocess.run(["pkill", "-f", "ollama serve"], check=False)
            time.sleep(1)
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(wait_s)
        except OSError:
            time.sleep(wait_s)
    assert last_err is not None
    raise last_err


def build_client(backend: str) -> tuple[OpenAI, str]:
    if backend == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
        return OpenAI(base_url=base_url, api_key="ollama"), model
    if backend == "teknofest":
        # Yarışmanın ortak API'si; adres ve model utils/config üzerinden env'den gelir
        from utils.config import vlm_endpoint

        ep = vlm_endpoint("teknofest")
        if not ep.base_url:
            raise SystemExit(
                "TEKNOFEST_BASE_URL (veya VLM_BASE_URL) tanımlı değil; .env dosyasını doldurun."
            )
        base_url = ep.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return OpenAI(base_url=base_url, api_key=ep.api_key or "teknofest"), ep.model
    if backend == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        model = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        api_key = os.getenv("OPENAI_API_KEY", "local")
        return OpenAI(base_url=base_url, api_key=api_key), model
    raise SystemExit(f"Bilinmeyen backend: {backend}")


def _call_vlm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_paths: list[Path],
    backend: str,
) -> str:
    if backend == "ollama":
        return ollama_chat(model, prompt, image_paths)
    max_side = int(os.getenv("MAX_IMAGE_SIDE", "768")) if backend == "teknofest" else None
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        b64 = encode_image(path, max_side=max_side)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    kwargs: dict = {}
    if backend == "openai":
        # Yerel vLLM / LM Studio kurulumları bu alanları kullanıyor
        kwargs["extra_body"] = {"options": {"num_ctx": 16384, "num_predict": 768}}
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=768,
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )
    return response.choices[0].message.content or ""


def _pick_second_look_frames(frames: list[dict], evidence: SceneEvidence) -> list[dict]:
    """Hareket tepelerine yakın 2–3 kare seç (token tasarrufu).

    Tek tepe yerine ilk üç tepe: ölçümde olay anını yakalama %25'ten %69'a çıkıyor
    (scripts/eval_wakeup.py).
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
    picked = ranked[:3]
    return sorted(picked, key=lambda f: float(f.get("t_sec", 0.0)))


def label_video(
    client: OpenAI,
    model: str,
    video_path: Path,
    frames_meta: dict,
    *,
    use_folder_hint: bool = True,
    backend: str = "ollama",
    evidence: SceneEvidence | None = None,
    use_second_look: bool = True,
    use_refine: bool = True,
) -> dict:
    frames = frames_meta["frames"]
    folder_cat = video_path.parent.name
    if evidence is None:
        evidence = analyze_video(video_path)

    frame_times = [frame["time"] for frame in frames]
    times = ", ".join(frame_times)
    numbered = "\n".join(
        f"{i}. kare zamanı {frame['time']} (events[].time için aday)"
        for i, frame in enumerate(frames, start=1)
    )
    hint = ""
    if use_folder_hint and folder_cat in {"normal", "near_miss", "accident"}:
        hint = f"Klasör ipucu (yanlış olabilir, gördüğüne göre düzelt): {folder_cat}\n"

    prompt = (
        f"Süre: {frames_meta['duration_sec']} saniye\n"
        f"Kare zamanları (events[].time SADECE bunlardan biri): {times}\n"
        f"Gönderilen kare sayısı: {len(frames)}\n"
        f"{hint}"
        f"{evidence.prompt_block()}\n"
        "Görseller aşağıda 1. kareden son kareye sıralı. "
        "İlk kare sakin olsa bile tüm diziyi oku; "
        "çarpışma, düşme, yanma, devrilme veya yerde kişi varsa onu yaz "
        "(accident + Yüksek). Sadece tehlikeli yaklaşma varsa near_miss.\n"
        f"{numbered}\n\n"
        + load_prompt()
    )
    image_paths = [Path(frame["path"]) for frame in frames]
    raw = _call_vlm(client, model, prompt, image_paths, backend)
    parsed = parse_json(raw)

    parsed_cat = parsed.get("category", "")
    if parsed_cat in {"normal", "near_miss", "accident"}:
        category = parsed_cat
    elif use_folder_hint and folder_cat in {"normal", "near_miss", "accident"}:
        category = folder_cat
    else:
        category = "normal"

    events = dedupe_events(snap_events_to_frame_times(parsed.get("events") or [], frame_times))

    label = {
        "video_id": frames_meta["video_id"],
        "filename": video_path.name,
        "category": category,
        "duration_sec": frames_meta["duration_sec"],
        "status": "auto",
        "labeled_by": f"qwen:{model}",
        "summary": parsed.get("summary", ""),
        "events": events,
        "risk": parsed.get("risk", "Orta"),
        "actions": parsed.get("actions", []),
        "notes": "Otomatik taslak. Gold değil; Streamlit'te kontrol edin.",
        "evidence": evidence.to_dict(),
    }

    # Yeni yöntem: sensör şüpheli + model sakin → kısa ikinci bakış
    if use_second_look and needs_second_look(label, evidence):
        focus = _pick_second_look_frames(frames, evidence)
        focus_paths = [Path(f["path"]) for f in focus]
        focus_times = ", ".join(f["time"] for f in focus)
        second_prompt = (
            f"{SECOND_LOOK_PROMPT}\n"
            f"{evidence.prompt_block()}\n"
            f"Odak kareler: {focus_times}\n"
            f"Önceki JSON özeti: risk={label.get('risk')}, "
            f"summary={label.get('summary')}\n\n"
            + load_prompt()
        )
        try:
            raw2 = _call_vlm(client, model, second_prompt, focus_paths, backend)
            parsed2 = parse_json(raw2)
            for key in ("summary", "events", "risk", "actions", "category"):
                if parsed2.get(key) not in (None, "", []):
                    label[key] = parsed2[key]
            focus_times_list = [f["time"] for f in focus]
            label["events"] = dedupe_events(
                snap_events_to_frame_times(
                    label.get("events") or [], focus_times_list or frame_times
                )
            )
            label["notes"] = (
                str(label.get("notes") or "") + " | ikinci bakış uygulandı"
            ).strip(" |")
        except Exception as exc:
            label["notes"] = (
                str(label.get("notes") or "") + f" | ikinci bakış atlandı: {exc}"
            ).strip(" |")

    # use_refine=False: büyük modelde kural katmanının katkısını ölçmek için ablasyon
    return refine_label(label, evidence) if use_refine else label


def competition_view(label: dict) -> dict:
    """Şartname kalıbı (utils.label_json ile aynı davranış)."""
    return label_to_spec(label)


def already_gold(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "gold"
    except json.JSONDecodeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/videos", type=Path)
    parser.add_argument("--frames", default="data/frames", type=Path)
    parser.add_argument("--out", default="data/labels", type=Path)
    parser.add_argument(
        "--backend",
        choices=["ollama", "teknofest", "openai"],
        default="ollama",
        help="teknofest = yarışmanın ortak API'si (.env: TEKNOFEST_BASE_URL, VLM_MODEL)",
    )
    parser.add_argument("--every-sec", default=0.75, type=float)
    parser.add_argument("--max-frames", default=6, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Yeni üretilecek taslak sayısı (0 = hepsi)")
    parser.add_argument("--only", default="", help="Dosya adı parçası, örn. Forklift")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = iter_videos(args.videos)
    if args.only:
        videos = [v for v in videos if args.only.lower() in v.name.lower()]
    if not videos:
        raise SystemExit(f"Video bulunamadı: {args.videos.resolve()}")

    client, model = build_client(args.backend)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "_auto_label_log.txt"
    print(f"Backend={args.backend} model={model} video={len(videos)}")

    ok = fail = skip = 0
    for i, video in enumerate(videos, start=1):
        if args.limit and ok >= args.limit:
            print(f"Limit ({args.limit}) doldu, duruyor.")
            break
        vid = safe_id(video)
        dest = args.out / f"{vid}.json"
        if dest.exists() and not args.overwrite:
            if already_gold(dest):
                print(f"[{i}/{len(videos)}] GOLD, dokunulmadı: {video.name}")
                skip += 1
                continue
            print(f"[{i}/{len(videos)}] taslak var, atlandı: {video.name}")
            skip += 1
            continue
        print(f"[{i}/{len(videos)}] etiketleniyor: {video.name}")
        try:
            frames_meta = extract_video(video, args.frames, args.every_sec, args.max_frames)
            label = label_video(client, model, video, frames_meta, backend=args.backend)
            dest.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
            spec_dest = args.out / f"{vid}_spec.json"
            spec_dest.write_text(
                json.dumps(competition_view(label), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  -> {dest.name} | {label.get('category')} | {label.get('risk')}")
            ok += 1
        except Exception as exc:
            fail += 1
            msg = f"HATA {video.name}: {exc}\n{traceback.format_exc()}\n"
            print(f"  HATA: {exc}")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(msg)

    print(f"Bitti. yeni={ok} atlanan={skip} hata={fail}")


if __name__ == "__main__":
    main()
