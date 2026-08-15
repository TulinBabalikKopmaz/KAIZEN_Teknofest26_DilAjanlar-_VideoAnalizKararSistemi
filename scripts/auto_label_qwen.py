#!/usr/bin/env python3
"""Qwen-VL ile şartname JSON'una uygun taslak etiket üretir.

Çıktı gold değil, taslaktır. Mevcut gold dosyaların üzerine yazmaz.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from extract_frames import extract_video, iter_videos, safe_id

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "video_label_prompt.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON bulunamadı: {text[:400]}")
    return json.loads(match.group(0))


def build_client(backend: str) -> tuple[OpenAI, str]:
    if backend == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
        return OpenAI(base_url=base_url, api_key="ollama"), model
    if backend == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        model = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        api_key = os.getenv("OPENAI_API_KEY", "local")
        return OpenAI(base_url=base_url, api_key=api_key), model
    raise SystemExit(f"Bilinmeyen backend: {backend}")


def label_video(client: OpenAI, model: str, video_path: Path, frames_meta: dict) -> dict:
    frames = frames_meta["frames"]
    folder_cat = video_path.parent.name
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Video dosyası: {video_path.name}\n"
                f"Klasör kategorisi (ipucu, yanlışsa düzelt): {folder_cat}\n"
                f"Süre: {frames_meta['duration_sec']} saniye\n"
                f"Gönderilen kare sayısı: {len(frames)}\n\n"
                + load_prompt()
            ),
        }
    ]
    for frame in frames:
        content.append({"type": "text", "text": f"Kare zamanı: {frame['time']}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(Path(frame['path']))}"
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "user", "content": content}],
        extra_body={"options": {"num_ctx": 16384, "num_predict": 512}},
    )
    parsed = parse_json(response.choices[0].message.content)

    category = folder_cat if folder_cat in {"normal", "near_miss", "accident"} else parsed.get(
        "category", "normal"
    )
    return {
        "video_id": frames_meta["video_id"],
        "filename": video_path.name,
        "category": category,
        "duration_sec": frames_meta["duration_sec"],
        "status": "auto",
        "labeled_by": f"qwen:{model}",
        "summary": parsed.get("summary", ""),
        "events": parsed.get("events", []),
        "risk": parsed.get("risk", "Orta"),
        "actions": parsed.get("actions", []),
        "notes": "Otomatik taslak. Gold değil; Streamlit'te kontrol edin.",
    }


def competition_view(label: dict) -> dict:
    return {
        "summary": label["summary"],
        "events": [
            {"time": e.get("time", "00:00"), "event": e.get("event", "")}
            for e in label.get("events", [])
            if e.get("event")
        ],
        "risk": label["risk"],
        "actions": label["actions"],
    }


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
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--every-sec", default=2.0, type=float)
    parser.add_argument("--max-frames", default=4, type=int)
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
            label = label_video(client, model, video, frames_meta)
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
