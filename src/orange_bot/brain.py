from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import BrainConfig
from .detectors import Detection
from .state import MinigameState
from .templates import TemplateMatch
from .vision import Target


Point = Tuple[int, int]


@dataclass(frozen=True)
class BrainTarget:
    center: Point
    score: float
    support_count: int
    primary_confidence: float
    secondary_confidence: float
    color_verify_ratio: float
    template_score: float
    color_area: float
    color_circularity: float


@dataclass
class _Cluster:
    center_x: float
    center_y: float
    primary_confidence: float = 0.0
    secondary_confidence: float = 0.0
    color_verify_ratio: float = 0.0
    color_area: float = 0.0
    color_circularity: float = 0.0
    template_score: float = 0.0
    support_count: int = 0

    def center(self) -> Point:
        return (int(round(self.center_x)), int(round(self.center_y)))


class MainBrain:
    def __init__(self, config: BrainConfig):
        self.config = config

    def choose_targets(
        self,
        color_targets: Sequence[Target],
        primary_detections: Sequence[Detection],
        secondary_detections: Sequence[Detection],
        template_matches: Sequence[TemplateMatch],
        minigame_state: MinigameState,
    ) -> List[BrainTarget]:
        if not minigame_state.active:
            return []

        clusters = self._build_clusters(
            color_targets=color_targets,
            primary_detections=primary_detections,
            secondary_detections=secondary_detections,
            template_matches=template_matches,
        )
        fused_targets: List[BrainTarget] = []

        for cluster in clusters:
            score = self._score_cluster(cluster) * max(minigame_state.confidence, 0.45)
            has_primary = cluster.primary_confidence > 0
            has_secondary = cluster.secondary_confidence > 0
            has_yolo = has_primary or has_secondary
            has_dual_yolo = has_primary and has_secondary
            has_template = cluster.template_score > 0
            strong_color = (
                cluster.color_verify_ratio >= 0.24
                or (
                    cluster.color_area >= 120
                    and cluster.color_verify_ratio >= 0.16
                )
                or (
                    cluster.color_circularity >= 0.22
                    and cluster.color_area >= 90
                    and cluster.color_verify_ratio >= 0.12
                )
            )

            if self.config.require_yolo_support and not has_yolo:
                if not self.config.allow_color_only_fallback:
                    continue
                if cluster.color_verify_ratio <= 0 and cluster.template_score <= 0:
                    continue

            if not has_yolo and not has_template and not strong_color:
                continue

            if score < self.config.min_click_score:
                continue

            if has_yolo and not has_dual_yolo and cluster.color_verify_ratio <= 0 and cluster.template_score <= 0:
                continue

            fused_targets.append(
                BrainTarget(
                    center=cluster.center(),
                    score=score,
                    support_count=cluster.support_count,
                    primary_confidence=cluster.primary_confidence,
                    secondary_confidence=cluster.secondary_confidence,
                    color_verify_ratio=cluster.color_verify_ratio,
                    template_score=cluster.template_score,
                    color_area=cluster.color_area,
                    color_circularity=cluster.color_circularity,
                )
            )

        fused_targets.sort(
            key=lambda target: (
                -target.score,
                -target.support_count,
                target.center[1],
                target.center[0],
            )
        )
        return fused_targets[: self.config.max_targets]

    def _build_clusters(
        self,
        color_targets: Sequence[Target],
        primary_detections: Sequence[Detection],
        secondary_detections: Sequence[Detection],
        template_matches: Sequence[TemplateMatch],
    ) -> List[_Cluster]:
        clusters: List[_Cluster] = []
        for detection in primary_detections:
            self._merge_detection(clusters, detection, "primary")
        for detection in secondary_detections:
            self._merge_detection(clusters, detection, "secondary")
        for color_target in color_targets:
            self._merge_color_target(clusters, color_target)
        for template_match in template_matches:
            self._merge_template_match(clusters, template_match)
        return clusters

    def _merge_detection(self, clusters: List[_Cluster], detection: Detection, source: str) -> None:
        cluster = self._find_cluster(clusters, detection.center)
        if cluster is None:
            cluster = _Cluster(center_x=detection.center[0], center_y=detection.center[1])
            clusters.append(cluster)
        self._update_center(cluster, detection.center)
        if source == "primary":
            cluster.primary_confidence = max(cluster.primary_confidence, detection.confidence)
        else:
            cluster.secondary_confidence = max(cluster.secondary_confidence, detection.confidence)
        cluster.support_count += 1

    def _merge_color_target(self, clusters: List[_Cluster], color_target: Target) -> None:
        cluster = self._find_cluster(clusters, color_target.center)
        if cluster is None:
            cluster = _Cluster(center_x=color_target.center[0], center_y=color_target.center[1])
            clusters.append(cluster)
        self._update_center(cluster, color_target.center)
        cluster.color_verify_ratio = max(cluster.color_verify_ratio, color_target.verify_ratio)
        cluster.color_area = max(cluster.color_area, color_target.area)
        cluster.color_circularity = max(cluster.color_circularity, color_target.circularity)
        cluster.support_count += 1

    def _merge_template_match(self, clusters: List[_Cluster], template_match: TemplateMatch) -> None:
        cluster = self._find_cluster(clusters, template_match.center)
        if cluster is None:
            cluster = _Cluster(center_x=template_match.center[0], center_y=template_match.center[1])
            clusters.append(cluster)
        self._update_center(cluster, template_match.center)
        cluster.template_score = max(cluster.template_score, template_match.score)
        cluster.support_count += 1

    def _find_cluster(self, clusters: Sequence[_Cluster], point: Point) -> _Cluster | None:
        max_distance = self.config.cluster_distance
        for cluster in clusters:
            dx = point[0] - cluster.center_x
            dy = point[1] - cluster.center_y
            if (dx * dx + dy * dy) ** 0.5 <= max_distance:
                return cluster
        return None

    @staticmethod
    def _update_center(cluster: _Cluster, point: Point) -> None:
        cluster.center_x = (cluster.center_x + point[0]) / 2.0
        cluster.center_y = (cluster.center_y + point[1]) / 2.0

    def _score_cluster(self, cluster: _Cluster) -> float:
        score = 0.0
        score += cluster.primary_confidence * self.config.primary_weight
        score += cluster.secondary_confidence * self.config.secondary_weight
        score += cluster.color_verify_ratio * self.config.color_weight
        score += cluster.template_score * self.config.template_weight
        if cluster.color_area >= 120 and cluster.color_verify_ratio >= 0.16:
            score += 0.45
        if cluster.color_verify_ratio >= 0.30:
            score += 0.35
        if cluster.color_circularity >= 0.22 and 90 <= cluster.color_area <= 8000:
            score += 0.65
        if cluster.primary_confidence > 0 and cluster.secondary_confidence > 0:
            score += self.config.dual_model_bonus
        if cluster.color_verify_ratio > 0 and (cluster.primary_confidence > 0 or cluster.secondary_confidence > 0):
            score += self.config.color_confirmation_bonus
        if cluster.template_score > 0 and (
            cluster.primary_confidence > 0
            or cluster.secondary_confidence > 0
            or cluster.color_verify_ratio > 0
        ):
            score += self.config.template_confirmation_bonus
        return score
