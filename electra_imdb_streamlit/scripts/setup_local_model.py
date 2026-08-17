"""Safely extract the trained ELECTRA ZIP into this project's model folder."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer_config.json"}
PROJECT_DIR = Path(__file__).resolve().parents[1]


def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract files while rejecting absolute paths and ../ path traversal."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(destination)


def find_model_root(extraction_dir: Path) -> Path:
    """Support ZIPs whose files are at the root or inside one folder."""
    if REQUIRED_FILES.issubset({path.name for path in extraction_dir.iterdir() if path.is_file()}):
        return extraction_dir
    candidates = [path.parent for path in extraction_dir.rglob("model.safetensors")]
    for candidate in candidates:
        if REQUIRED_FILES.issubset({path.name for path in candidate.iterdir() if path.is_file()}):
            return candidate
    raise FileNotFoundError(
        "The ZIP does not contain config.json, model.safetensors, and tokenizer_config.json together."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="Path to ELECTRA-base_IMDb_deployment.zip")
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_DIR / "model",
        help="Target model folder (default: project/model)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace model files already present in the destination.",
    )
    args = parser.parse_args()

    zip_path = args.zip.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not zip_path.is_file():
        print(f"Error: ZIP not found: {zip_path}", file=sys.stderr)
        return 1

    existing_model = destination / "model.safetensors"
    if existing_model.exists() and not args.overwrite:
        print(
            f"Error: {existing_model} already exists. Use --overwrite only if you intend to replace it.",
            file=sys.stderr,
        )
        return 1

    temporary_dir = destination.parent / f".{destination.name}_extracting"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    try:
        safe_extract(zip_path, temporary_dir)
        model_root = find_model_root(temporary_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source in model_root.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    print(f"Model installed successfully in: {destination}")
    print("Next command: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

