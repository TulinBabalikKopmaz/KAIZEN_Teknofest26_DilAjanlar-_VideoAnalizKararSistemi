"""Jüri arayüzü: Claude Design HTML'i + gerçek analiz API.

    py app/jury_server.py

Tarayıcı: http://127.0.0.1:8503
Maket (sahte veri): http://127.0.0.1:8503/claude-mock
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from aiohttp import web
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).resolve().parent / "jury_ui"
DESKTOP_MOCK = Path.home() / "Desktop" / "KAIZEN Juri Demo Arayuzu.html"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from utils import config  # noqa: E402
from utils import demo_pipeline as kz_pipeline  # noqa: E402
from utils import display as kz_display  # noqa: E402
from utils.live_watch import StreamConfig, get_hub, looks_like_jpeg  # noqa: E402

INCOMING = ROOT / "data" / "incoming"
VIDEO_DIRS = (ROOT / "data" / "videos", INCOMING)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _media_url(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(str(path))
    try:
        resolved = raw.resolve()
    except OSError:
        return ""
    try:
        rel = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return ""
    return "/media/" + rel.as_posix()


def _pack_result(result: kz_pipeline.DemoResult, *, backup: bool = False) -> dict[str, Any]:
    result = kz_pipeline.polish_demo_result(result)
    spec = result.spec or {}
    v = kz_display.verdict(result.label.get("category"), spec.get("risk"))
    source = kz_display.model_source(result.provider, result.model_calls, backup=backup)
    hard = kz_display.hard_case_note(result.label, spec, result.evidence)
    law = kz_display.law_support_card(getattr(result, "law_detail", "") or "")
    frames: list[dict[str, str]] = []
    for frame in result.frames or []:
        path = frame.get("demo_path") or frame.get("path")
        frames.append(
            {
                "time": str(frame.get("time") or ""),
                "url": _media_url(str(path) if path else ""),
            }
        )
    video = Path(result.video) if result.video else None
    return {
        "ok": True,
        "backup": backup,
        "video_url": _media_url(video) if video else "",
        "video_name": video.name if video else "",
        "answer": result.answer or "",
        "summary": spec.get("summary") or "",
        "situation": v["situation"],
        "decision": v["decision"],
        "subtitle": v["subtitle"],
        "tone": v["tone"],
        "kicker": v["kicker"],
        "source": source,
        "total_s": round(float(result.total_s or 0), 1),
        "events": list(spec.get("events") or []),
        "actions": list(spec.get("actions") or []),
        "frames": frames,
        "spec": spec,
        "timings": result.timings or {},
        "model_calls": result.model_calls or [],
        "evidence": result.evidence or {},
        "law": law,
        "law_note": getattr(result, "law_note", "") or "",
        "hard": hard,
        "warnings": list(result.warnings or []),
        "spec_footnote": kz_display.spec_footnote(),
        "provider": result.provider,
    }


def _apply_provider(lock: bool, provider: str) -> None:
    if lock:
        os.environ["PROVIDER"] = "teknofest"
        os.environ["FALLBACK_PROVIDER"] = "none"
        return
    os.environ.pop("FALLBACK_PROVIDER", None)
    os.environ["PROVIDER"] = provider if provider in config.PROVIDERS else "teknofest"


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(UI / "index.html")


async def asset(request: web.Request) -> web.FileResponse:
    name = request.match_info["name"]
    allowed = {"styles.css", "layout.css", "app.js"}
    if name not in allowed:
        raise web.HTTPNotFound()
    return web.FileResponse(UI / name)


async def claude_mock(_request: web.Request) -> web.StreamResponse:
    if DESKTOP_MOCK.is_file():
        return web.FileResponse(DESKTOP_MOCK)
    raise web.HTTPNotFound(
        text="Claude HTML masaüstünde yok: KAIZEN Juri Demo Arayuzu.html"
    )


def _list_videos() -> list[dict[str, str]]:
    found: list[Path] = []
    for directory in VIDEO_DIRS:
        if directory.exists():
            found.extend(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, str]] = []
    root = ROOT.resolve()
    for path in found:
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        out.append({"name": path.name, "rel": rel.as_posix()})
    return out


def _live_pack() -> dict[str, Any]:
    hub = get_hub()
    snap = hub.snapshot()
    snap["ok"] = True
    snap["running"] = hub.running()
    snap["source_info"] = kz_display.model_source(str(snap.get("provider") or ""), backup=False)
    law_detail = str(snap.get("law_detail") or "")
    snap["law"] = kz_display.law_support_card(law_detail) if law_detail else None
    return snap


def _restart_hub(cfg: StreamConfig) -> None:
    hub = get_hub()
    if hub.running():
        hub.stop()
    hub.start(cfg)


async def boot(_request: web.Request) -> web.Response:
    runs = [
        {"name": path.name, "mtime": path.stat().st_mtime}
        for path in kz_pipeline.list_saved_runs(limit=12)
    ]
    return web.json_response(
        {
            "describe": config.describe(),
            "fast": config.demo_fast_mode(),
            "frames": config.demo_max_frames(),
            "prompt": kz_pipeline.DEFAULT_PROMPT,
            "runs": runs,
            "videos": _list_videos(),
        }
    )


async def analyze(request: web.Request) -> web.Response:
    post = await request.post()
    upload = post.get("video")
    prompt = str(post.get("prompt") or kz_pipeline.DEFAULT_PROMPT)
    lock = str(post.get("lock") or "1") != "0"
    provider = str(post.get("provider") or "teknofest")
    fast = str(post.get("fast") or "0") == "1"
    rag = str(post.get("rag") or "1") != "0"
    try:
        frames = int(str(post.get("frames") or "8"))
    except ValueError:
        frames = 8
    frames = max(4, min(12, frames))
    if upload is None or not getattr(upload, "filename", ""):
        return web.json_response({"ok": False, "error": "Önce bir video yükleyin."}, status=400)
    suffix = Path(str(upload.filename)).suffix.lower()
    if suffix not in VIDEO_EXTS:
        return web.json_response({"ok": False, "error": "mp4 / mov / avi / mkv / webm"}, status=400)
    INCOMING.mkdir(parents=True, exist_ok=True)
    dest = INCOMING / Path(str(upload.filename)).name
    dest.write_bytes(upload.file.read())
    _apply_provider(lock, provider)

    def _run() -> kz_pipeline.DemoResult:
        return kz_pipeline.run_demo_analysis_sync(
            dest,
            prompt,
            max_frames=frames,
            fast=fast,
            use_rag=rag and not fast,
        )

    try:
        result = await asyncio.get_running_loop().run_in_executor(None, _run)
        payload = _pack_result(result, backup=False)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)[:400]}, status=500)
    return web.json_response(payload)


async def load_run(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    chosen = next((path for path in kz_pipeline.list_saved_runs(limit=24) if path.name == name), None)
    if chosen is None:
        return web.json_response({"ok": False, "error": "Yedek bulunamadı."}, status=404)
    result = kz_pipeline.load_saved_run(chosen)
    return web.json_response(_pack_result(result, backup=True))


async def media(request: web.Request) -> web.StreamResponse:
    rel = request.match_info["path"]
    target = (ROOT / Path(rel)).resolve()
    root = ROOT.resolve()
    if root not in target.parents and target != root:
        raise web.HTTPForbidden()
    if not target.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(target)


async def live_start(request: web.Request) -> web.Response:
    post = await request.post()
    kind = str(post.get("kind") or "dosya").strip().lower()
    lock = str(post.get("lock") or "1") != "0"
    provider = str(post.get("provider") or "teknofest")
    try:
        motion = float(str(post.get("motion") or "12"))
    except ValueError:
        motion = 12.0
    try:
        cooldown = float(str(post.get("cooldown") or "8"))
    except ValueError:
        cooldown = 8.0
    _apply_provider(lock, provider)
    source = "0"
    loop_file = False
    if kind == "webcam":
        cam = str(post.get("cam") or "0").strip()
        source = cam if cam.isdigit() else "0"
    else:
        upload = post.get("video")
        if upload is None or not getattr(upload, "filename", ""):
            return web.json_response({"ok": False, "error": "Önce bir video seçin."}, status=400)
        suffix = Path(str(upload.filename)).suffix.lower()
        if suffix not in VIDEO_EXTS:
            return web.json_response({"ok": False, "error": "mp4 / mov / avi / mkv / webm"}, status=400)
        INCOMING.mkdir(parents=True, exist_ok=True)
        dest = INCOMING / Path(str(upload.filename)).name
        dest.write_bytes(upload.file.read())
        source = str(dest)
        loop_file = True
    cfg = StreamConfig(
        source=source,
        motion_threshold=max(4.0, min(40.0, motion)),
        cooldown_s=max(4.0, min(20.0, cooldown)),
        loop_file=loop_file,
        max_workers=1,
    )
    try:
        await asyncio.get_running_loop().run_in_executor(None, _restart_hub, cfg)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)[:400]}, status=500)
    return web.json_response(_live_pack())


async def live_stop(_request: web.Request) -> web.Response:
    hub = get_hub()
    await asyncio.get_running_loop().run_in_executor(None, hub.stop)
    return web.json_response(_live_pack())


async def live_status(_request: web.Request) -> web.Response:
    return web.json_response(_live_pack())


async def live_preview(_request: web.Request) -> web.StreamResponse:
    data = get_hub().latest_jpeg()
    headers = {"Cache-Control": "no-store"}
    if looks_like_jpeg(data):
        return web.Response(body=data, content_type="image/jpeg", headers=headers)
    return web.Response(status=204, headers=headers)


def build_app() -> web.Application:
    app = web.Application(client_max_size=1024 * 1024 * 400)
    app.router.add_get("/", index)
    app.router.add_get("/claude-mock", claude_mock)
    app.router.add_get("/api/boot", boot)
    app.router.add_post("/api/analyze", analyze)
    app.router.add_get("/api/run/{name}", load_run)
    app.router.add_post("/api/live/start", live_start)
    app.router.add_post("/api/live/stop", live_stop)
    app.router.add_get("/api/live/status", live_status)
    app.router.add_get("/api/live/preview", live_preview)
    app.router.add_get("/media/{path:.+}", media)
    app.router.add_get("/{name}", asset)
    return app


def main() -> None:
    app = build_app()
    print("Jüri arayüzü  http://127.0.0.1:8503")
    print("Claude maket   http://127.0.0.1:8503/claude-mock")
    web.run_app(app, host="127.0.0.1", port=8503)


if __name__ == "__main__":
    main()
