import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orange_bot.config import CustomDetectorConfig
from orange_bot.custom_detector import CustomOrangeDetector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom ONNX click detector on a folder of screenshots.")
    parser.add_argument("--images-dir", default=".", help="Directory with screenshots.")
    parser.add_argument("--model", default="models/orange_click.onnx", help="ONNX model path.")
    parser.add_argument("--output-dir", default="eval_click_model", help="Directory for debug output.")
    parser.add_argument("--confidence", type=float, default=0.18, help="Center confidence threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.18, help="Mask threshold.")
    return parser.parse_args()


def read_image(image_path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def iter_images(images_dir: Path) -> list[Path]:
    return [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def save_debug_image(image_bgr: np.ndarray, detections, output_path: Path) -> None:
    debug = image_bgr.copy()
    for detection in detections:
        x, y = detection.center
        cv2.circle(debug, (x, y), 16, (0, 255, 0), 2)
        cv2.putText(debug, f"{detection.confidence:.2f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(output_path), debug)


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = CustomOrangeDetector(
        CustomDetectorConfig(
            name="click_model",
            model_path=args.model,
            confidence_threshold=args.confidence,
            mask_threshold=args.mask_threshold,
        )
    )

    results = []
    for image_path in iter_images(images_dir):
        image_bgr = read_image(image_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections, _ = detector.detect_with_debug(image_rgb)
        save_debug_image(image_bgr, detections, output_dir / f"{image_path.stem}_debug.png")
        results.append(
            {
                "image": image_path.name,
                "count": len(detections),
                "points": [
                    {"x": detection.center[0], "y": detection.center[1], "score": round(detection.confidence, 4)}
                    for detection in detections
                ],
            }
        )
        print(f"{image_path.name} detections={len(detections)}")

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
