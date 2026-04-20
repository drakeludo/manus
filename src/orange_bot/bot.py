import time
import winsound
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import keyboard

from .brain import BrainTarget, MainBrain
from .config import BotConfig, DEFAULT_CONFIG
from .custom_detector import CustomOrangeDetector
from .detectors import Detection, YoloDetector
from .state import MinigameState, MinigameStateTracker
from .templates import TemplateDetector, TemplateMatch
from .vision import OrangeVision, Target
from .win32_input import click_many, key_down, key_press, key_up


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


@dataclass(frozen=True)
class AnalysisFrame:
    color_targets: list[Target]
    primary_detections: list[Detection]
    secondary_detections: list[Detection]
    template_matches: list[TemplateMatch]
    start_matches: list[TemplateMatch]
    minigame_state: MinigameState


class OrangeBot:
    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.vision = OrangeVision(config.vision)
        self.primary_detector = CustomOrangeDetector(config.custom_primary)
        self.secondary_detector = CustomOrangeDetector(config.custom_secondary)
        self.legacy_primary_detector = YoloDetector(config.yolo_primary)
        self.legacy_secondary_detector = YoloDetector(config.yolo_secondary)
        self.template_detector = TemplateDetector(config.templates)
        self.state_tracker = MinigameStateTracker(config.state)
        self.brain = MainBrain(config.brain)
        self.running = False
        self.stopped = False
        self.recent_clicks: deque[tuple[float, tuple[int, int]]] = deque()
        self._detector_status_logged = False

    def beep(self, frequency: int, duration: int) -> None:
        winsound.Beep(frequency, duration)

    def run_cycle(self) -> None:
        controls = self.config.controls
        self.state_tracker.reset()

        log("Перезарядка дерева (отход и возврат)")
        self.rearm()

        log("Активация дерева")
        key_press(controls.interact_key)
        time.sleep(controls.interact_wait)

        total_clicked = 0
        cycle_started_at = time.time()
        empty_scan_streak = 0

        while total_clicked < controls.max_clicks_per_tree:
            if not self.running or self.stopped:
                return
            if time.time() - cycle_started_at >= controls.max_cycle_seconds:
                break

            self._log_detector_status_once()
            targets, state = self._scan_burst()
            if not state.active:
                empty_scan_streak += 1
                log(f"Мини-игра не подтверждена: {state.reason}")
                if empty_scan_streak >= controls.empty_scan_streak_to_rearm:
                    break
                time.sleep(controls.retry_delay)
                continue

            if not targets:
                empty_scan_streak += 1
                if empty_scan_streak >= controls.empty_scan_streak_to_rearm:
                    break
                time.sleep(controls.retry_delay)
                continue

            empty_scan_streak = 0
            log(f"Найдено целей: {len(targets)} | state={state.reason} ({state.confidence:.2f})")
            for target in self._order_targets_for_burst(targets):
                if not self.running or self.stopped:
                    return
                if self._is_recent_click(target):
                    continue

                click_many(
                    self._get_click_points(target.center),
                    settle_delay=controls.mouse_settle_delay,
                    press_delay=controls.mouse_press_delay,
                )
                if controls.click_cooldown > 0:
                    time.sleep(controls.click_cooldown)
                self._remember_click(target)
                total_clicked += 1
                if total_clicked >= controls.max_clicks_per_tree:
                    break

            time.sleep(controls.post_batch_delay)

        log(f"Цикл завершен. Клики: {total_clicked}")

    def rearm(self) -> None:
        controls = self.config.controls
        for step_index, (key, duration, label) in enumerate(self._get_rearm_steps()):
            log(label)
            self._move_exact(key, duration)
            if step_index == 0:
                time.sleep(controls.trigger_settle_delay)
        time.sleep(controls.post_rearm_delay)

    def toggle(self) -> None:
        self.running = not self.running
        if self.running:
            self.state_tracker.reset()
        log("Бот включен" if self.running else "Бот на паузе")
        self.beep(800 if self.running else 400, 150)

    def stop(self) -> None:
        self.stopped = True
        self.running = False
        log("Выход")

    def main_loop(self) -> None:
        controls = self.config.controls
        log("Готово. F6 - старт/пауза, F7 - выход")
        self.beep(1000, 200)

        while not self.stopped:
            if keyboard.is_pressed("f6"):
                self.toggle()
                time.sleep(controls.toggle_debounce)

            if keyboard.is_pressed(controls.screenshot_key):
                self.save_screenshot()
                time.sleep(controls.toggle_debounce)

            if keyboard.is_pressed("f7"):
                self.stop()
                break

            if self.running:
                try:
                    self.run_cycle()
                except Exception as error:
                    log(f"Ошибка в цикле: {error}")
                    self.running = False

            time.sleep(controls.loop_sleep)

    def save_screenshot(self) -> None:
        path = Path(self.config.controls.dataset_path)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"capture_{int(time.time())}.png"
        full_path = path / filename
        frame = self.vision.capture_image()
        cv2.imwrite(str(full_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        log(f"Скриншот сохранен: {full_path}")

    def _remember_click(self, target: BrainTarget) -> None:
        self._prune_recent_clicks()
        self.recent_clicks.append((time.time(), target.center))

    def _is_recent_click(self, target: BrainTarget) -> bool:
        self._prune_recent_clicks()
        radius = self.config.controls.click_position_radius
        target_x, target_y = target.center
        for _, (saved_x, saved_y) in self.recent_clicks:
            dx = target_x - saved_x
            dy = target_y - saved_y
            if (dx * dx + dy * dy) ** 0.5 < radius:
                return True
        return False

    def _prune_recent_clicks(self) -> None:
        expires_after = self.config.controls.click_position_ttl
        now = time.time()
        while self.recent_clicks and now - self.recent_clicks[0][0] > expires_after:
            self.recent_clicks.popleft()

    def _scan_burst(self) -> tuple[list[BrainTarget], MinigameState]:
        controls = self.config.controls
        burst_targets: list[BrainTarget] = []
        burst_color_targets: list[Target] = []
        last_state = MinigameState(active=False, confidence=0.0, reason="none")
        for _ in range(controls.scan_burst_count):
            analysis = self._analyze_frame()
            last_state = analysis.minigame_state
            burst_color_targets.extend(analysis.color_targets)
            burst_targets.extend(
                self.brain.choose_targets(
                    color_targets=analysis.color_targets,
                    primary_detections=analysis.primary_detections,
                    secondary_detections=analysis.secondary_detections,
                    template_matches=analysis.template_matches,
                    minigame_state=analysis.minigame_state,
                )
            )
            if controls.scan_burst_delay > 0:
                time.sleep(controls.scan_burst_delay)
        merged_targets = self._merge_burst_targets(burst_targets)
        merged_color_targets = self._merge_color_targets(burst_color_targets)
        expanded_targets = self._expand_targets_for_coverage(merged_targets, merged_color_targets)
        return expanded_targets, last_state

    def _analyze_frame(self) -> AnalysisFrame:
        frame = self.vision.capture_image()
        color_targets = self.vision.find_targets(frame)
        primary_detections = self.primary_detector.detect(frame)
        secondary_detections = self.secondary_detector.detect(frame)
        if self.config.yolo_primary.enabled:
            primary_detections.extend(self.legacy_primary_detector.detect(frame))
        if self.config.yolo_secondary.enabled:
            secondary_detections.extend(self.legacy_secondary_detector.detect(frame))
        template_matches = self.template_detector.detect_oranges(frame)
        start_matches = self.template_detector.detect_start(frame)
        minigame_state = self.state_tracker.update(
            yolo_count=len(primary_detections) + len(secondary_detections),
            color_count=len(color_targets),
            template_count=len(template_matches),
            start_template_count=len(start_matches),
        )
        return AnalysisFrame(
            color_targets=color_targets,
            primary_detections=primary_detections,
            secondary_detections=secondary_detections,
            template_matches=template_matches,
            start_matches=start_matches,
            minigame_state=minigame_state,
        )

    def _merge_burst_targets(self, targets: list[BrainTarget]) -> list[BrainTarget]:
        merged: list[BrainTarget] = []
        for target in sorted(targets, key=lambda item: item.score, reverse=True):
            if any(
                self._distance(target.center, saved.center) < self.config.controls.click_position_radius
                for saved in merged
            ):
                continue
            merged.append(target)
            if len(merged) >= self.config.controls.max_clicks_per_tree:
                break
        return merged

    def _merge_color_targets(self, targets: list[Target]) -> list[Target]:
        merged: list[Target] = []
        min_spacing = self.config.vision.target.min_target_spacing
        for target in sorted(targets, key=lambda item: (item.verify_ratio, item.area), reverse=True):
            if any(self._distance(target.center, saved.center) < min_spacing for saved in merged):
                continue
            merged.append(target)
        return merged

    def _expand_targets_for_coverage(
        self,
        base_targets: list[BrainTarget],
        color_targets: list[Target],
    ) -> list[BrainTarget]:
        if not self.config.controls.coverage_fill_enabled:
            return base_targets

        selected = list(base_targets)
        if len(selected) >= self.config.controls.max_clicks_per_tree or not color_targets:
            return selected[: self.config.controls.max_clicks_per_tree]

        covered_indexes = {
            index
            for index, color_target in enumerate(color_targets)
            if any(self._covers_color_target(saved.center, color_target) for saved in selected)
        }

        while len(selected) < self.config.controls.max_clicks_per_tree and len(covered_indexes) < len(color_targets):
            best_index = None
            best_gain: list[int] = []
            best_target: Target | None = None
            for index, color_target in enumerate(color_targets):
                if any(
                    self._distance(color_target.center, saved.center) < self.config.controls.click_position_radius
                    for saved in selected
                ):
                    continue
                newly_covered = [
                    color_index
                    for color_index, candidate in enumerate(color_targets)
                    if color_index not in covered_indexes and self._covers_color_target(color_target.center, candidate)
                ]
                if not newly_covered:
                    continue
                if (
                    len(newly_covered) > len(best_gain)
                    or (
                        len(newly_covered) == len(best_gain)
                        and best_target is not None
                        and (color_target.verify_ratio, color_target.area)
                        > (best_target.verify_ratio, best_target.area)
                    )
                    or best_target is None
                ):
                    best_index = index
                    best_gain = newly_covered
                    best_target = color_target

            if best_index is None or best_target is None:
                break

            selected.append(self._brain_target_from_color(best_target, len(best_gain)))
            covered_indexes.update(best_gain)

        return selected[: self.config.controls.max_clicks_per_tree]

    def _covers_color_target(self, click_center: tuple[int, int], color_target: Target) -> bool:
        effect_radius = self.config.controls.click_effect_radius
        for point in self._get_click_points(click_center):
            if self._distance(point, color_target.center) <= effect_radius:
                return True
        return False

    @staticmethod
    def _brain_target_from_color(target: Target, support_count: int) -> BrainTarget:
        return BrainTarget(
            center=target.center,
            score=float(target.verify_ratio) + support_count * 0.01,
            support_count=support_count,
            primary_confidence=0.0,
            secondary_confidence=0.0,
            color_verify_ratio=target.verify_ratio,
            template_score=0.0,
            color_area=target.area,
            color_circularity=target.circularity,
        )

    def _get_click_points(self, center: tuple[int, int]) -> list[tuple[int, int]]:
        controls = self.config.controls
        if not controls.click_spread_enabled:
            return [center]

        center_x, center_y = center
        points: list[tuple[int, int]] = []
        for offset_x, offset_y in controls.click_spread_pattern:
            point = (center_x + offset_x, center_y + offset_y)
            if point not in points:
                points.append(point)
        return points

    def _get_rearm_steps(self) -> list[tuple[str, float, str]]:
        controls = self.config.controls
        return [
            (controls.back_key, controls.trigger_back_duration, "Шаг назад"),
            (controls.forward_key, controls.trigger_forward_duration, "Шаг вперед (возврат)"),
        ]

    @staticmethod
    def _order_targets_for_burst(targets: list[BrainTarget]) -> list[BrainTarget]:
        return sorted(targets, key=lambda target: (target.center[1], target.center[0]))

    @staticmethod
    def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _move_exact(key: str, duration: float) -> None:
        deadline = time.perf_counter() + duration
        key_down(key)
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.005))
        finally:
            key_up(key)

    def _log_detector_status_once(self) -> None:
        if self._detector_status_logged:
            return
        self._detector_status_logged = True
        primary_status = "ready" if self.primary_detector.is_ready else self.primary_detector.disabled_reason
        secondary_status = "ready" if self.secondary_detector.is_ready else self.secondary_detector.disabled_reason
        legacy_primary_status = (
            "ready" if self.legacy_primary_detector.is_ready else self.legacy_primary_detector.disabled_reason
        )
        legacy_secondary_status = (
            "ready" if self.legacy_secondary_detector.is_ready else self.legacy_secondary_detector.disabled_reason
        )
        log(f"Custom primary: {primary_status}")
        log(f"Custom secondary: {secondary_status}")
        log(f"Legacy YOLO primary: {legacy_primary_status}")
        log(f"Legacy YOLO secondary: {legacy_secondary_status}")
