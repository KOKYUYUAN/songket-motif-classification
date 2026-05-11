"""Split tampuk_manggis images into train/val (80/20).
Drop your 53 original images into raw_data/tampuk_manggis before running.
"""
from pathlib import Path
import random
import shutil

TRAIN_RATIO = 0.8


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_path = project_root / "raw_data" / "tampuk_manggis"
    train_path = project_root / "dataset" / "train" / "tampuk_manggis"
    val_path = project_root / "dataset" / "val" / "tampuk_manggis"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise SystemExit(f"Source folder missing: {source_path}")

    files = [f for f in source_path.iterdir() if f.is_file()]
    if not files:
        raise SystemExit(f"No files found in {source_path}. Add images first.")

    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for src in train_files:
        shutil.copy(src, train_path / src.name)
    for src in val_files:
        shutil.copy(src, val_path / src.name)

    print(f"✅ Done! {len(train_files)} images in Train, {len(val_files)} in Val.")


if __name__ == "__main__":
    main()
