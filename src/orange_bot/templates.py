from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .config import TemplateConfig


Point = Tuple[int, int]


@dataclass(frozen=True)
class TemplateMatch:
    center: Point
    score: float
    source: str


class TemplateDetector:
    def __init__(self, config: TemplateConfig):
        self.config = config

    def detect_oranges(self, frame_rgb: np.ndarray) -> List[TemplateMatch]:
        template_paths = self._resolve_template_paths(
            self.config.orange_template_glob,
            self.config.orange_template_path,
        )
        matches: List[TemplateMatch] = []
        for template_path in template_paths:
            matches.extend(
                self._detect(
                    frame_rgb=frame_rgb,
                    template_path=template_path,
                    threshold=self.config.orange_threshold,
                    scales=self.config.orange_scales,
                    source=f"template_orange:{template_path.name}",
                )
            )
        return self._dedupe(matches)

    def detect_start(self, frame_rgb: np.ndarray) -> List[TemplateMatch]:
        return self._detect(
            frame_rgb=frame_rgb,
            template_path=Path(self.config.start_template_path),
            threshold=self.config.start_threshold,
            scales=self.config.start_scales,
            source="template_start",
        )

    def _detect(
        self,
        frame_rgb: np.ndarray,
        template_path: Path,
        threshold: float,
        scales: Iterable[float],
        source: str,
    ) -> List[TemplateMatch]:
        if not self.config.enabled:
            return []
        if not template_path.exists():
            return []

        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            return []

        screen_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        matches: List[TemplateMatch] = []
        for scale in scales:
            resized = cv2.resize(template, None, fx=scale, fy=scale)
            if resized.size == 0:
                continue
            if resized.shape[0] > screen_gray.shape[0] or resized.shape[1] > screen_gray.shape[1]:
                continue

            result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            height, width = resized.shape
            for point in zip(*locations[::-1]):
                center = (int(point[0] + width / 2), int(point[1] + height / 2))
                matches.append(
                    TemplateMatch(
                        center=center,
                        score=float(result[point[1], point[0]]),
                        source=source,
                    )
                )

        return matches

    def _resolve_template_paths(self, glob_pattern: str, fallback_path: str) -> List[Path]:
        globbed = sorted(Path().glob(glob_pattern))
        if globbed:
            return globbed
        return [Path(fallback_path)]

    def _dedupe(self, matches: List[TemplateMatch]) -> List[TemplateMatch]:
        deduped: List[TemplateMatch] = []
        matches = sorted(matches, key=lambda match: match.score, reverse=True)
        for match in matches:
            if any(self._distance(match.center, saved.center) < self.config.dedupe_distance for saved in deduped):
                continue
            deduped.append(match)
        return deduped

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)
