from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import CustomDetectorConfig
from .detectors import Detection


@dataclass(frozen=True)
class DetectorDebug:
    heatmap: np.ndarray
    mask: np.ndarray


class CustomOrangeDetector:
    def __init__(self, config: CustomDetectorConfig):
        self.config = config
        self._session = None
        self._load_attempted = False
        self._disabled_reason: Optional[str] = None
        self._input_name: Optional[str] = None
        self._output_names: list[str] = []

    @property
    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._session is not None

    @property
    def disabled_reason(self) -> Optional[str]:
        self._ensure_loaded()
        return self._disabled_reason

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        detections, _ = self.detect_with_debug(frame_rgb)
        return detections

    def detect_with_debug(self, frame_rgb: np.ndarray) -> tuple[List[Detection], DetectorDebug | None]:
        self._ensure_loaded()
        if self._session is None:
            return [], None

        original_height, original_width = frame_rgb.shape[:2]
        input_tensor = self._preprocess(frame_rgb)
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        heatmap, mask = self._decode_outputs(outputs)
        detections = self._postprocess_maps(
            heatmap=heatmap,
            mask=mask,
            original_width=original_width,
            original_height=original_height,
        )
        debug = DetectorDebug(heatmap=heatmap, mask=mask)
        return detections, debug

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
            import onnxruntime as ort
        except Exception as error:
            self._disabled_reason = f"onnxruntime import failed: {error}"
            return

        try:
            self._session = ort.InferenceSession(str(model_path), providers=list(self.config.providers))
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [output.name for output in self._session.get_outputs()]
        except Exception as error:
            self._disabled_reason = f"onnx load failed: {error}"
            self._session = None

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame_rgb,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _decode_outputs(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        output_map = {name: value for name, value in zip(self._output_names, outputs)}
        heatmap = output_map.get(self.config.heatmap_output_name)
        mask = output_map.get(self.config.mask_output_name)

        if heatmap is None:
            heatmap = self._first_spatial_output(outputs)
        if mask is None:
            mask = self._second_spatial_output(outputs, heatmap)

        heatmap_2d = self._squeeze_to_2d(heatmap)
        mask_2d = self._squeeze_to_2d(mask) if mask is not None else np.zeros_like(heatmap_2d, dtype=np.float32)

        heatmap_2d = self._sigmoid_if_needed(heatmap_2d)
        mask_2d = self._sigmoid_if_needed(mask_2d)
        return heatmap_2d.astype(np.float32), mask_2d.astype(np.float32)

    def _postprocess_maps(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> List[Detection]:
        if heatmap.size == 0:
            return []

        kernel_size = max(3, int(self.config.peak_kernel_size) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pooled = cv2.dilate(heatmap, kernel, iterations=1)
        peak_mask = (heatmap >= self.config.confidence_threshold) & (heatmap >= pooled - 1e-6)

        ys, xs = np.where(peak_mask)
        candidates: list[tuple[float, tuple[int, int]]] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            heat_score = float(heatmap[y, x])
            mask_score = float(mask[y, x]) if mask.size else 0.0
            if mask_score < self.config.mask_threshold:
                continue
            score = heat_score + mask_score * self.config.mask_weight
            point = (
                int(round((x / max(1, heatmap.shape[1] - 1)) * max(0, original_width - 1))),
                int(round((y / max(1, heatmap.shape[0] - 1)) * max(0, original_height - 1))),
            )
            candidates.append((score, point))

        candidates.sort(key=lambda item: item[0], reverse=True)
        detections: list[Detection] = []
        for score, center in candidates:
            if any(self._distance(center, saved.center) < self.config.min_peak_distance for saved in detections):
                continue
            detections.append(
                Detection(
                    center=center,
                    bbox=(center[0], center[1], center[0], center[1]),
                    confidence=float(min(1.0, score)),
                    source=self.config.name,
                )
            )
            if len(detections) >= self.config.max_detections:
                break
        return detections

    @staticmethod
    def _squeeze_to_2d(array: np.ndarray | None) -> np.ndarray:
        if array is None:
            return np.zeros((0, 0), dtype=np.float32)
        squeezed = np.asarray(array)
        while squeezed.ndim > 2:
            squeezed = squeezed[0]
        return squeezed.astype(np.float32)

    @staticmethod
    def _sigmoid_if_needed(array: np.ndarray) -> np.ndarray:
        if array.size == 0:
            return array
        min_value = float(array.min())
        max_value = float(array.max())
        if 0.0 <= min_value and max_value <= 1.0:
            return array
        return 1.0 / (1.0 + np.exp(-array))

    @staticmethod
    def _first_spatial_output(outputs: list[np.ndarray]) -> np.ndarray:
        spatial = [np.asarray(output) for output in outputs if np.asarray(output).ndim >= 2]
        if not spatial:
            return np.zeros((0, 0), dtype=np.float32)
        spatial.sort(key=lambda item: np.asarray(item).size, reverse=True)
        return spatial[0]

    @staticmethod
    def _second_spatial_output(outputs: list[np.ndarray], heatmap: np.ndarray | None) -> np.ndarray | None:
        heatmap_id = id(heatmap) if heatmap is not None else None
        spatial = [np.asarray(output) for output in outputs if np.asarray(output).ndim >= 2 and id(output) != heatmap_id]
        if not spatial:
            return None
        spatial.sort(key=lambda item: np.asarray(item).size, reverse=True)
        return spatial[0]

    @staticmethod
    def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)
