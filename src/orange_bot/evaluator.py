import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .brain import BrainTarget, MainBrain
from .config import BotConfig, DEFAULT_CONFIG
from .custom_detector import CustomOrangeDetector
from .detectors import YoloDetector
from .state import MinigameStateTracker
from .templates import TemplateDetector
from .vision import OrangeVision


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageEvaluation:
    image_path: str
    predicted_targets: int
    state_active: bool
    state_confidence: float
    state_reason: str
    matched_targets: int | None = None
    false_positives: int | None = None
    missed_targets: int | None = None
    precision: float | None = None
    recall: float | None = None


class OrangeBotEvaluator:
    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.vision = OrangeVision(config.vision)
        self.primary_detector = CustomOrangeDetector(config.custom_primary)
        self.secondary_detector = CustomOrangeDetector(config.custom_secondary)
        self.legacy_primary_detector = YoloDetector(config.yolo_primary)
        self.legacy_secondary_detector = YoloDetector(config.yolo_secondary)
        self.template_detector = TemplateDetector(config.templates)
        self.brain = MainBrain(config.brain)

    def evaluate_directory(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        label_dir: str | Path | None = None,
    ) -> list[ImageEvaluation]:
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        label_dir = Path(label_dir) if label_dir else None

        results: list[ImageEvaluation] = []
        for image_path in self._iter_images(image_dir):
            evaluation = self.evaluate_image(image_path, output_dir, label_dir)
            results.append(evaluation)

        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return results

    def evaluate_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        label_dir: str | Path | None = None,
    ) -> ImageEvaluation:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        label_dir = Path(label_dir) if label_dir else None

        frame_bgr = self._read_image(image_path)
        if frame_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        color_targets = self.vision.find_targets(frame_rgb)
        primary_detections = self.primary_detector.detect(frame_rgb)
        secondary_detections = self.secondary_detector.detect(frame_rgb)
        if self.config.yolo_primary.enabled:
            primary_detections.extend(self.legacy_primary_detector.detect(frame_rgb))
        if self.config.yolo_secondary.enabled:
            secondary_detections.extend(self.legacy_secondary_detector.detect(frame_rgb))
        template_matches = self.template_detector.detect_oranges(frame_rgb)
        start_matches = self.template_detector.detect_start(frame_rgb)

        state_tracker = MinigameStateTracker(self.config.state)
        minigame_state = state_tracker.update(
            yolo_count=len(primary_detections) + len(secondary_detections),
            color_count=len(color_targets),
            template_count=len(template_matches),
            start_template_count=len(start_matches),
        )

        brain_targets = self.brain.choose_targets(
            color_targets=color_targets,
            primary_detections=primary_detections,
            secondary_detections=secondary_detections,
            template_matches=template_matches,
            minigame_state=minigame_state,
        )

        labels = self._load_labels(image_path, label_dir)
        matched_targets = false_positives = missed_targets = None
        precision = recall = None
        if labels is not None:
            matched_targets, false_positives, missed_targets = self._match_points(
                predicted=[target.center for target in brain_targets],
                expected=labels,
                radius=self.config.controls.click_position_radius * 2,
            )
            precision = matched_targets / max(1, matched_targets + false_positives)
            recall = matched_targets / max(1, matched_targets + missed_targets)

        self._save_debug_image(
            frame_bgr=frame_bgr,
            brain_targets=brain_targets,
            color_count=len(color_targets),
            primary_count=len(primary_detections),
            secondary_count=len(secondary_detections),
            template_count=len(template_matches),
            minigame_state=minigame_state,
            output_path=output_dir / f"{image_path.stem}_debug.png",
            labels=labels,
        )

        return ImageEvaluation(
            image_path=str(image_path),
            predicted_targets=len(brain_targets),
            state_active=minigame_state.active,
            state_confidence=minigame_state.confidence,
            state_reason=minigame_state.reason,
            matched_targets=matched_targets,
            false_positives=false_positives,
            missed_targets=missed_targets,
            precision=precision,
            recall=recall,
        )

    @staticmethod
    def _iter_images(image_dir: Path) -> Iterable[Path]:
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    @staticmethod
    def _read_image(image_path: Path) -> np.ndarray | None:
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    @staticmethod
    def _load_labels(image_path: Path, label_dir: Path | None) -> list[tuple[int, int]] | None:
        if label_dir is None:
            return None
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            return None
        data = json.loads(label_path.read_text(encoding="utf-8"))
        return [(int(item["x"]), int(item["y"])) for item in data.get("points", [])]

    @staticmethod
    def _match_points(
        predicted: list[tuple[int, int]],
        expected: list[tuple[int, int]],
        radius: int,
    ) -> tuple[int, int, int]:
        remaining_expected = expected.copy()
        matched = 0
        for point in predicted:
            best_index = None
            best_distance = None
            for index, expected_point in enumerate(remaining_expected):
                distance = ((point[0] - expected_point[0]) ** 2 + (point[1] - expected_point[1]) ** 2) ** 0.5
                if distance <= radius and (best_distance is None or distance < best_distance):
                    best_distance = distance
                    best_index = index
            if best_index is not None:
                matched += 1
                remaining_expected.pop(best_index)
        false_positives = max(0, len(predicted) - matched)
        missed_targets = len(remaining_expected)
        return matched, false_positives, missed_targets

    @staticmethod
    def _save_debug_image(
        frame_bgr: np.ndarray,
        brain_targets: list[BrainTarget],
        color_count: int,
        primary_count: int,
        secondary_count: int,
        template_count: int,
        minigame_state,
        output_path: Path,
        labels: list[tuple[int, int]] | None,
    ) -> None:
        debug = frame_bgr.copy()
        for target in brain_targets:
            x, y = target.center
            cv2.circle(debug, (x, y), 16, (0, 255, 0), 2)
            cv2.putText(
                debug,
                f"{target.score:.2f}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if labels:
            for x, y in labels:
                cv2.circle(debug, (x, y), 12, (0, 0, 255), 2)

        lines = [
            f"state={minigame_state.active} {minigame_state.reason} {minigame_state.confidence:.2f}",
            f"brain={len(brain_targets)} color={color_count} model1={primary_count} model2={secondary_count} tpl={template_count}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                debug,
                line,
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imwrite(str(output_path), debug)
