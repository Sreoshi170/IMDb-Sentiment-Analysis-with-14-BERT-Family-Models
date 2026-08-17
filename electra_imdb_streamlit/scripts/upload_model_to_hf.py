"""Upload the trained ELECTRA ZIP to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer_config.json"}


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(destination)


def find_model_root(extraction_dir: Path) -> Path:
    for weights_path in [extraction_dir / "model.safetensors", *extraction_dir.rglob("model.safetensors")]:
        candidate = weights_path.parent
        names = {path.name for path in candidate.iterdir() if path.is_file()}
        if REQUIRED_FILES.issubset(names):
            return candidate
    raise FileNotFoundError("Required Transformers model files were not found in the ZIP.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", required=True, type=Path, help="Path to the downloaded model ZIP")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. username/electra-imdb")
    parser.add_argument("--private", action="store_true", help="Create a private model repository")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_zip.is_file():
        raise FileNotFoundError(f"Model ZIP not found: {args.model_zip}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    with tempfile.TemporaryDirectory(prefix="electra_imdb_upload_") as temp_name:
        extraction_dir = Path(temp_name)
        safe_extract(args.model_zip.resolve(), extraction_dir)
        model_root = find_model_root(extraction_dir)

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"Uploading model files to https://huggingface.co/{args.repo_id} ...")
        api.upload_folder(
            folder_path=model_root,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload fine-tuned ELECTRA IMDb sentiment model",
        )

    print("Upload complete.")
    print(f'Add this to Streamlit secrets: HF_MODEL_REPO = "{args.repo_id}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

