from pathlib import Path

import cv2

from .config import BotConfig, DEFAULT_CONFIG
from .vision import OrangeVision


class TemplateBootstrapper:
    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.vision = OrangeVision(config.vision)

    def build_from_directory(self, image_dir: str | Path = ".") -> tuple[list[Path], Path | None]:
        image_dir = Path(image_dir)
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() == ".png")
        orange_template_paths = self._build_orange_templates(images, models_dir)
        start_template_path = self._build_start_template(images, models_dir / "minigame_start.png")
        return orange_template_paths, start_template_path

    def _build_orange_templates(self, images: list[Path], models_dir: Path) -> list[Path]:
        candidates: list[tuple[float, object, int, int]] = []
        for image_path in images:
            frame_bgr = cv2.imread(str(image_path))
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            targets = self.vision.find_targets(frame_rgb)
            for target in targets:
                x, y = target.center
                if target.circularity < 0.70:
                    continue
                if target.area < 1800 or target.area > 4200:
                    continue
                if y > frame_bgr.shape[0] * 0.65:
                    continue
                score = target.verify_ratio + target.circularity + (target.area / 5000.0)
                candidates.append((score, frame_bgr, x, y))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        saved_centers: list[tuple[int, int]] = []
        output_paths: list[Path] = []
        for index, (_, frame_bgr, x, y) in enumerate(candidates):
            if any(((x - sx) ** 2 + (y - sy) ** 2) ** 0.5 < 80 for sx, sy in saved_centers):
                continue
            radius = 42
            top = max(0, y - radius)
            bottom = min(frame_bgr.shape[0], y + radius)
            left = max(0, x - radius)
            right = min(frame_bgr.shape[1], x + radius)
            crop = frame_bgr[top:bottom, left:right]
            output_path = models_dir / f"orange_{len(output_paths) + 1}.png"
            cv2.imwrite(str(output_path), crop)
            output_paths.append(output_path)
            saved_centers.append((x, y))
            if len(output_paths) >= 6:
                break

        if output_paths:
            cv2.imwrite(str(models_dir / "orange.png"), cv2.imread(str(output_paths[0])))
        return output_paths

    @staticmethod
    def _build_start_template(images: list[Path], output_path: Path) -> Path | None:
        if not images:
            return None
        frame_bgr = cv2.imread(str(images[0]))
        if frame_bgr is None:
            return None
        crop = frame_bgr[18:74, 18:360]
        cv2.imwrite(str(output_path), crop)
        return output_path
