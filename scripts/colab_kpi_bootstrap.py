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
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check)


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


def link_data(repo: Path, drive_root: Path) -> None:
    """Drive'daki data/ klasörünü repo data/ ile birleştirir."""
    drive_data = drive_root / "data"
    repo_data = repo / "data"
    repo_data.mkdir(parents=True, exist_ok=True)

    def bind(name: str, *, copy_gold_first: bool = False) -> None:
        src = drive_data / name
        dest = repo_data / name
        src.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            if copy_gold_first and dest.is_dir():
                for f in dest.glob("gold_labels*.json"):
                    target = src / f.name
                    if not target.exists():
                        shutil.copy2(f, target)
                        print(f"  kopyalandı → Drive: {target}")
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.symlink_to(src, target_is_directory=True)
        print(f"  bağlandı: {dest} → {src}")

    for name in ("videos", "predictions_wide", "frames", "labels"):
        bind(name)
    bind("exports", copy_gold_first=True)

    gold = drive_data / "exports" / "gold_labels_hepsi.json"
    if not gold.exists():
        print(
            f"UYARI: gold yok. Mac'ten yükleyin:\n"
            f"  {drive_data / 'exports' / 'gold_labels_hepsi.json'}"
        )
    else:
        print("  gold OK")

    videos = drive_data / "videos"
    n = len(list(videos.rglob("*.mp4"))) if videos.exists() else 0
    print(f"  video sayısı: {n}")


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
            "ultralytics",
            "orjson",
        ]
    )


def install_ollama(model: str) -> None:
    # Colab'de Ollama kurulum scripti zstd ister
    run(["bash", "-c", "apt-get update -qq && apt-get install -y -qq zstd > /dev/null"])
    if shutil.which("ollama") is None:
        run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
    # Sunucuyu arka planda başlat
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import time

    time.sleep(3)
    run(["ollama", "pull", model], check=False)


def write_env(repo: Path, model: str) -> None:
    env = repo / ".env"
    env.write_text(
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
    args = parser.parse_args()

    print("=== TEKNOFEST Colab KPI bootstrap ===")
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
    print("\nHazır. Sonraki hücrede KPI çalıştırın:")
    print(
        f"  python scripts/run_kpi_wide.py --n 18 --seed 42 --model {args.model} "
        f"--pred-dir data/predictions_wide --no-second-look"
    )


if __name__ == "__main__":
    main()
