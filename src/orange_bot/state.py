from dataclasses import dataclass

from .config import StateConfig


@dataclass(frozen=True)
class MinigameState:
    active: bool
    confidence: float
    reason: str


class MinigameStateTracker:
    def __init__(self, config: StateConfig):
        self.config = config
        self._active_frames = 0
        self._inactive_frames = 0
        self._active = False

    def reset(self) -> None:
        self._active_frames = 0
        self._inactive_frames = 0
        self._active = False

    def update(
        self,
        yolo_count: int,
        color_count: int,
        template_count: int,
        start_template_count: int,
    ) -> MinigameState:
        evidence_score = 0
        reasons: list[str] = []

        if yolo_count >= self.config.yolo_presence_threshold:
            evidence_score += 2
            reasons.append("yolo")
        if color_count >= self.config.color_presence_threshold:
            evidence_score += 1
            reasons.append("color")
        if template_count >= self.config.template_presence_threshold:
            evidence_score += 1
            reasons.append("template")
        if start_template_count > 0:
            evidence_score += 2
            reasons.append("start")

        has_active_signal = evidence_score >= self.config.min_targets_for_active
        if has_active_signal:
            self._active_frames += 1
            self._inactive_frames = 0
        else:
            self._inactive_frames += 1
            self._active_frames = 0

        if self._active_frames >= self.config.required_active_frames:
            self._active = True
        elif self._inactive_frames >= self.config.required_inactive_frames:
            self._active = False

        confidence = min(1.0, evidence_score / 4.0)
        return MinigameState(
            active=self._active,
            confidence=confidence,
            reason=",".join(reasons) if reasons else "none",
        )
