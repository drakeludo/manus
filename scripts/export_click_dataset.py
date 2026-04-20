import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orange_bot.config import DEFAULT_CONFIG
from orange_bot.vision import OrangeVision


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pseudo-labeled click dataset from Majestic screenshots.")
    parser.add_argument("--images-dir", default=".", help="Directory with source screenshots.")
    parser.add_argument("--output-dir", default="datasets/click_oranges", help="Output dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-verify", type=float, default=0.16, help="Minimum color verify ratio.")
    parser.add_argument("--min-area", type=float, default=80.0, help="Minimum contour area.")
    return parser.parse_args()


def read_image(image_path: Path):
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def iter_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name.startswith("2026-")
    ]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    vision = OrangeVision(DEFAULT_CONFIG.vision)

    images = iter_images(images_dir)
    if not images:
        raise SystemExit(f"No source images found in {images_dir}")

    train_images_dir = output_dir / "images" / "train"
    val_images_dir = output_dir / "images" / "val"
    train_labels_dir = output_dir / "labels" / "train"
    val_labels_dir = output_dir / "labels" / "val"
    for directory in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shuffled = images[:]
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_names = {path.name for path in shuffled[:val_count]}

    exported_images = 0
    exported_points = 0
    for image_path in images:
        frame_bgr = read_image(image_path)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        targets = [
            target
            for target in vision.find_targets(frame_rgb)
            if target.verify_ratio >= args.min_verify and target.area >= args.min_area
        ]
        if not targets:
            continue

        is_val = image_path.name in val_names
        image_out_dir = val_images_dir if is_val else train_images_dir
        label_out_dir = val_labels_dir if is_val else train_labels_dir
        shutil.copy2(image_path, image_out_dir / image_path.name)

        label = {
            "image": image_path.name,
            "points": [
                {
                    "x": int(target.center[0]),
                    "y": int(target.center[1]),
                    "radius": int(max(8, min(24, round((target.area / np.pi) ** 0.5)))),
                    "verify_ratio": round(float(target.verify_ratio), 4),
                    "area": round(float(target.area), 2),
                }
                for target in targets
            ],
        }
        (label_out_dir / f"{image_path.stem}.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exported_images += 1
        exported_points += len(label["points"])

    print(f"exported_images={exported_images}")
    print(f"exported_points={exported_points}")
    print(f"dataset_dir={output_dir}")


if __name__ == "__main__":
    main()
