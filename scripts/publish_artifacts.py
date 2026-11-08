"""List or upload the four published artifacts with checksums."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ARTIFACTS = (
    "lfm3-steer.onnx",
    "steer_stats.bin",
    "catalog.json",
    "heatmap.bin",
    "recipes.json",
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(directory: Path) -> list[dict]:
    rows = []
    for name in ARTIFACTS:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"missing artifact {path}")
        rows.append({"name": name, "size": path.stat().st_size, "sha256": checksum(path)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("web/public"))
    parser.add_argument("--repo", default=None, help="Hugging Face repo id to upload to")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rows = collect(args.dir)
    for row in rows:
        print(f"{row['name']}\t{row['size']}\t{row['sha256']}")

    if args.dry_run or not args.repo:
        print("dry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    for row in rows:
        api.upload_file(
            path_or_fileobj=str(args.dir / row["name"]),
            path_in_repo=row["name"],
            repo_id=args.repo,
            revision=args.revision,
        )
        print(f"uploaded {row['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
