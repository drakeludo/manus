from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import YoloModelConfig


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    center: Point
    bbox: BBox
    confidence: float
    source: str


class YoloDetector:
    def __init__(self, config: YoloModelConfig):
        self.config = config
        self._model = None
        self._load_attempted = False
        self._disabled_reason: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def disabled_reason(self) -> Optional[str]:
        self._ensure_loaded()
        return self._disabled_reason

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        self._ensure_loaded()
        if self._model is None:
            return []

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            source=frame_bgr,
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != self.config.target_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                detections.append(
                    Detection(
                        center=center,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf[0].item()),
                        source=self.config.name,
                    )
                )
        return detections

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self.config.enabled:
            self._disabled_reason = "disabled"
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            self._disabled_reason = f"missing model: {model_path}"
            return

        try:
            from ultralytics import YOLO
        except Exception as error:
            self._disabled_reason = f"ultralytics import failed: {error}"
            return

        try:
            self._model = YOLO(str(model_path))
        except Exception as error:
            self._disabled_reason = f"model load failed: {error}"
