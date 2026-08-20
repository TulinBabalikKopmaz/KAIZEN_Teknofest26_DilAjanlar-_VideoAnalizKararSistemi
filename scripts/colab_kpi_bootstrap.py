#!/usr/bin/env python3
"""Colab / uzak GPU için KPI ortamını hazırlar.

Notebook'tan çağrılır:
  !python scripts/colab_kpi_bootstrap.py --drive-root /content/drive/MyDrive/KAIZEN_KPI
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
    )


def ensure_gpu() -> None:
    try:
        import torch

        ok = torch.cuda.is_available()
        print(f"CUDA: {ok}  device={torch.cuda.get_device_name(0) if ok else 'yok'}")
        if not ok:
            print(
                "UYARI: GPU yok. Runtime → Change runtime type → T4 GPU seçin, "
                "sonra hücreyi yeniden çalıştırın."
            )
    except ImportError:
        print("torch yok (şimdilik sorun değil; Ollama kendi GPU'sunu kullanır).")


GOLD_FROM_GIT = (
    "gold_labels_hepsi.json",
    "gold_labels_hepsi.jsonl",
    "gold_labels_hepsi_spec.json",
    "gold_labels_hepsi_ozet.csv",
    "splits.json",
)


def _snapshot_git_gold(repo_exports: Path) -> dict[str, Path]:
    """Symlink'ten önce git'teki gold/split dosyalarını temp'e alır.

    Drive'da eski gold varsa onu korumak Colab'i yanlış cevap anahtarına kilitler.
    Kaynak: GitHub'daki export. Videolar Drive'da kalır.
    """
    import tempfile

    if not repo_exports.exists() or repo_exports.is_symlink():
        return {}
    tmp = Path(tempfile.mkdtemp(prefix="kaizen_gold_"))
    kept: dict[str, Path] = {}
    for name in GOLD_FROM_GIT:
        src = repo_exports / name
        if src.is_file():
            dest = tmp / name
            shutil.copy2(src, dest)
            kept[name] = dest
    return kept


def link_data(repo: Path, drive_root: Path) -> None:
    """Drive'daki data/ klasörünü repo data/ ile birleştirir."""
    drive_data = drive_root / "data"
    repo_data = repo / "data"
    repo_data.mkdir(parents=True, exist_ok=True)
    git_gold = _snapshot_git_gold(repo_data / "exports")

    def bind(name: str) -> None:
        src = drive_data / name
        dest = repo_data / name
        src.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.symlink_to(src, target_is_directory=True)
        print(f"  bağlandı: {dest} → {src}")

    for name in ("videos", "predictions_wide", "frames", "labels", "exports"):
        bind(name)

    drive_exports = drive_data / "exports"
    for name, src in git_gold.items():
        target = drive_exports / name
        shutil.copy2(src, target)
        print(f"  git gold → Drive (üzerine yazıldı): {name}")

    gold = drive_exports / "gold_labels_hepsi.json"
    splits = drive_exports / "splits.json"
    if not gold.exists():
        print(
            f"UYARI: gold yok. PC'den yükleyin:\n"
            f"  {gold}"
        )
    else:
        print("  gold OK", gold)
    if splits.exists():
        print("  splits OK", splits)
    else:
        print("  UYARI: splits.json yok — evaluate_kpi --split dev çalışmaz")

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = drive_data / "videos"
    found = (
        [p for p in videos.rglob("*") if p.suffix.lower() in video_exts]
        if videos.exists()
        else []
    )
    print(f"  video sayısı: {len(found)}  ({videos})")
    for sub in ("accident", "near_miss", "normal"):
        d = videos / sub
        n_sub = (
            len([p for p in d.rglob("*") if p.suffix.lower() in video_exts])
            if d.exists()
            else 0
        )
        print(f"    {sub}/: {n_sub}")
    if not found:
        print(
            "UYARI: Drive'da video yok. PC'den mp4 yükleyin:\n"
            f"  {videos}/accident|near_miss|normal/\n"
            "  (gold dosyası tek başına yetmez; KPI videolara ihtiyaç duyar)"
        )


def install_python_deps() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "openai",
            "opencv-python-headless",
            "python-dotenv",
            "aiohttp",
            "ultralytics",
            "orjson",
        ]
    )


def wait_ollama(timeout_s: float = 90.0) -> None:
    url = "http://127.0.0.1:11434/api/tags"
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("  ollama API ayakta", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.5)
    raise SystemExit(
        f"Ollama API ayağa kalkmadı ({timeout_s:.0f}s). Son hata: {last_err}\n"
        "Yeni hücrede dene: !pkill ollama; !nohup ollama serve >/tmp/ollama.log 2>&1 &"
    )


def model_installed(model: str) -> bool:
    result = subprocess.run(
        ["ollama", "list"],
        check=False,
        text=True,
        capture_output=True,
    )
    out = (result.stdout or "") + (result.stderr or "")
    print(out, flush=True)
    # "qwen2.5vl:7b" veya satırda "qwen2.5vl" geçebilir
    name = model.split(":")[0]
    return name in out


def install_ollama(model: str) -> None:
    # Colab: zstd + pciutils (GPU tespiti için)
    run(
        [
            "bash",
            "-c",
            "apt-get update -qq && apt-get install -y -qq zstd pciutils > /dev/null",
        ]
    )
    if shutil.which("ollama") is None:
        run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])

    # Eski serve varsa bırak
    subprocess.run(["pkill", "-f", "ollama serve"], check=False)
    time.sleep(1)

    log_path = Path("/tmp/ollama_serve.log")
    log_f = log_path.open("w", encoding="utf-8")
    print("+ ollama serve (arka plan)", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_ollama()

    print(f"+ ollama pull {model}  ( indirme birkaç dk sürebilir; çıktı akacak )", flush=True)
    pull = subprocess.run(["ollama", "pull", model], check=False)
    if pull.returncode != 0:
        print(f"UYARI: ollama pull exit={pull.returncode}", flush=True)
        print("--- /tmp/ollama_serve.log (son 40 satır) ---", flush=True)
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-40:]), flush=True)

    print("+ ollama list", flush=True)
    if not model_installed(model):
        raise SystemExit(
            f"Model yok: {model}\n"
            f"Manuel dene:\n"
            f"  !ollama pull {model}\n"
            f"  !ollama list"
        )
    print(f"  model OK: {model}", flush=True)


def write_env(repo: Path, model: str) -> None:
    env = repo / ".env"
    env.write_text(
        f"PROVIDER=ollama\n"
        f"OLLAMA_BASE_URL=http://127.0.0.1:11434/v1\n"
        f"OLLAMA_MODEL={model}\n"
        f"OLLAMA_NUM_CTX=16384\n",
        encoding="utf-8",
    )
    print(f"  yazıldı: {env}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/KAIZEN_KPI"),
        help="Google Drive'daki KPI klasörü",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/content/teknofest-video-ajan"),
        help="Colab'de klonlanan repo yolu",
    )
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument(
        "--only-ollama",
        action="store_true",
        help="Sadece Ollama kur/çek (pip+Drive atla)",
    )
    args = parser.parse_args()

    print("=== TEKNOFEST Colab KPI bootstrap ===", flush=True)

    if args.only_ollama:
        ensure_gpu()
        install_ollama(args.model)
        print("\nOllama hazır.", flush=True)
        return

    ensure_gpu()
    args.drive_root.mkdir(parents=True, exist_ok=True)
    (args.drive_root / "data" / "videos").mkdir(parents=True, exist_ok=True)
    (args.drive_root / "data" / "exports").mkdir(parents=True, exist_ok=True)

    if not args.repo.exists():
        raise SystemExit(f"Repo yok: {args.repo} — önce git clone hücresini çalıştırın.")

    os.chdir(args.repo)
    install_python_deps()
    link_data(args.repo, args.drive_root)
    write_env(args.repo, args.model)
    if not args.skip_ollama:
        install_ollama(args.model)
    print("\nHazır. Sonraki hücrede KPI çalıştırın:", flush=True)
    print(
        f"  python scripts/run_kpi_wide.py --n all --split dev --model {args.model} "
        f"--pred-dir data/predictions_wide --tag goldv2 --no-second-look --resume",
        flush=True,
    )


if __name__ == "__main__":
    main()
