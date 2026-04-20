from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

from .config import VisionConfig


Point = Tuple[int, int]


@dataclass(frozen=True)
class Target:
    center: Point
    area: float
    fill_ratio: float
    circularity: float
    verify_ratio: float


class OrangeVision:
    def __init__(self, config: VisionConfig):
        self.config = config

    def capture_image(self) -> np.ndarray:
        region = self.config.region
        if self.config.screen.use_full_screen or (region.right <= region.left or region.bottom <= region.top):
            return np.array(ImageGrab.grab())
        return np.array(ImageGrab.grab(bbox=(region.left, region.top, region.right, region.bottom)))

    def get_targets(self) -> List[Target]:
        return self.find_targets(self.capture_image())

    def find_targets(self, image: np.ndarray) -> List[Target]:
        tree_regions = self._find_tree_regions(image)
        targets: List[Target] = []
        for offset_x, offset_y, roi in tree_regions:
            roi_targets = self._find_targets_in_roi(roi, offset_x, offset_y)
            targets.extend(roi_targets)
        targets.sort(
            key=lambda target: (
                -target.verify_ratio,
                -target.area,
                target.center[1],
                target.center[0],
            )
        )
        return self._deduplicate_targets(targets)

    def _find_tree_regions(self, image: np.ndarray) -> List[tuple[int, int, np.ndarray]]:
        if not self.config.screen.use_full_screen:
            return [(self.config.region.left, self.config.region.top, image)]

        orange_mask = self._build_mask(image)
        mask_u8 = orange_mask.astype(np.uint8) * 255
        if cv2.countNonZero(mask_u8) == 0:
            return [(0, 0, image)]

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        expansion = self.config.screen.tree_search_expansion
        image_height, image_width = image.shape[:2]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(float(self.config.target.min_pixels), 80.0):
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 12 or height < 12:
                continue
            left = max(0, x - expansion)
            top = max(0, y - expansion)
            right = min(image_width, x + width + expansion)
            bottom = min(image_height, y + height + expansion)
            candidates.append((area, (left, top, right, bottom)))

        if not candidates:
            return [(0, 0, image)]

        candidates.sort(key=lambda item: item[0], reverse=True)
        merged_boxes: list[tuple[int, int, int, int]] = []
        for _, box in candidates:
            if any(self._boxes_intersect(box, saved_box) for saved_box in merged_boxes):
                merged_boxes = [self._merge_boxes(box, saved_box) if self._boxes_intersect(box, saved_box) else saved_box for saved_box in merged_boxes]
            else:
                merged_boxes.append(box)
            if len(merged_boxes) >= self.config.screen.max_tree_candidates:
                break

        return [(left, top, image[top:bottom, left:right]) for left, top, right, bottom in merged_boxes]

    def _find_targets_in_roi(self, image: np.ndarray, offset_x: int, offset_y: int) -> List[Target]:
        mask = self._build_mask(image)
        point_count = int(mask.sum())
        if point_count < self.config.target.min_pixels:
            return []

        mask_u8 = (mask.astype(np.uint8)) * 255
        mask_u8 = cv2.medianBlur(mask_u8, 5)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets: List[Target] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.config.target.min_pixels < area < self.config.target.max_pixels:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < self.config.target.min_component_width or height < self.config.target.min_component_height:
                continue

            bbox_area = width * height
            fill_ratio = area / max(1.0, bbox_area)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (4.0 * np.pi * area) / max(1.0, perimeter * perimeter)
            aspect_ratio = max(width, height) / max(1.0, min(width, height))
            if aspect_ratio > self.config.target.max_aspect_ratio and fill_ratio > 0.45:
                continue

            roi = mask[y : y + height, x : x + width]
            extracted_centers = self._extract_centers_from_roi(roi)
            is_compact_component = circularity >= 0.55 and aspect_ratio <= 1.8
            if len(extracted_centers) >= 2:
                local_centers = extracted_centers
            elif is_compact_component:
                center = self._contour_center(contour)
                local_centers = [(center[0] - x, center[1] - y)]
            else:
                local_centers = list(extracted_centers)
                local_centers.extend(self._sample_points_from_mask(roi))
                local_centers = self._deduplicate_points(local_centers)
                if not local_centers:
                    center = self._contour_center(contour)
                    local_centers = [(center[0] - x, center[1] - y)]

            for roi_x, roi_y in local_centers[: self.config.target.max_targets_per_contour]:
                local_x = x + roi_x
                local_y = y + roi_y
                verify_ratio = self._verify_target(mask, local_x, local_y)
                if verify_ratio < self.config.target.min_verify_ratio * 0.5:
                    continue

                targets.append(
                    Target(
                        center=(local_x + offset_x, local_y + offset_y),
                        area=area,
                        fill_ratio=fill_ratio,
                        circularity=circularity,
                        verify_ratio=verify_ratio,
                    )
                )

        return targets

    def _sample_points_from_mask(self, roi_mask: np.ndarray) -> List[Point]:
        if roi_mask.size == 0:
            return []

        height, width = roi_mask.shape
        step = max(8, self.config.target.grid_step)
        sampled: List[Point] = []

        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                if not roi_mask[y, x]:
                    continue
                verify_ratio = self._verify_target(roi_mask, x, y)
                if verify_ratio < self.config.target.min_verify_ratio:
                    continue
                sampled.append((x, y))

        if sampled:
            return sampled

        ys, xs = np.nonzero(roi_mask)
        if len(xs) == 0:
            return []
        center_x = int(xs.mean())
        center_y = int(ys.mean())
        return [(center_x, center_y)]

    def _deduplicate_points(self, points: Sequence[Point]) -> List[Point]:
        deduped: List[Point] = []
        for point in points:
            if any(self._distance(point, saved) < self.config.target.min_target_spacing for saved in deduped):
                continue
            deduped.append(point)
        return deduped

    def _build_mask(self, image: np.ndarray) -> np.ndarray:
        orange = self.config.orange_rgb
        red = image[:, :, 0].astype(np.int16)
        green = image[:, :, 1].astype(np.int16)
        blue = image[:, :, 2].astype(np.int16)
        rgb_mask = (
            (red > orange.min_red)
            & ((red - green) > orange.min_rg_gap)
            & (blue < orange.max_blue)
            & (((red + green + blue) / 3) > orange.min_brightness)
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        primary_hsv_mask = (
            (hsv[:, :, 0] >= orange.hue_min)
            & (hsv[:, :, 0] <= orange.hue_max)
            & (hsv[:, :, 1] >= orange.min_saturation)
            & (hsv[:, :, 2] >= orange.min_value)
        )
        shadow_hsv_mask = (
            (hsv[:, :, 0] >= orange.shadow_hue_min)
            & (hsv[:, :, 0] <= orange.shadow_hue_max)
            & (hsv[:, :, 1] >= orange.shadow_min_saturation)
            & (hsv[:, :, 2] >= orange.shadow_min_value)
            & (red > green)
        )

        mask = rgb_mask | primary_hsv_mask | shadow_hsv_mask
        mask_u8 = mask.astype(np.uint8) * 255
        close_kernel = np.ones(
            (self.config.target.close_kernel_size, self.config.target.close_kernel_size),
            dtype=np.uint8,
        )
        open_kernel = np.ones(
            (self.config.target.open_kernel_size, self.config.target.open_kernel_size),
            dtype=np.uint8,
        )
        dilate_kernel = np.ones(
            (self.config.target.dilate_kernel_size, self.config.target.dilate_kernel_size),
            dtype=np.uint8,
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        if self.config.target.dilate_iterations > 0:
            mask_u8 = cv2.dilate(mask_u8, dilate_kernel, iterations=self.config.target.dilate_iterations)
        return mask_u8 > 0

    def _extract_centers_from_roi(self, roi_mask: np.ndarray) -> List[Point]:
        roi_u8 = (roi_mask.astype(np.uint8)) * 255
        distance = cv2.distanceTransform(roi_u8, cv2.DIST_L2, 5)
        max_distance = float(distance.max())
        if max_distance < self.config.target.min_peak_distance:
            return []

        distance_threshold = max(
            self.config.target.min_peak_distance,
            max_distance * self.config.target.peak_threshold_ratio,
        )
        dilated = cv2.dilate(distance, np.ones((3, 3), dtype=np.uint8), iterations=1)
        peak_mask = (distance >= distance_threshold) & (distance >= (dilated - 1e-6))
        peaks_u8 = peak_mask.astype(np.uint8) * 255

        count, _, stats, centroids = cv2.connectedComponentsWithStats(peaks_u8)
        candidates: List[tuple[float, Point]] = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] <= 0:
                continue
            center_x = int(round(float(centroids[index][0])))
            center_y = int(round(float(centroids[index][1])))
            peak_strength = float(distance[center_y, center_x])
            candidates.append((peak_strength, (center_x, center_y)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        centers: List[Point] = []
        for _, center in candidates:
            if any(self._distance(center, saved_center) < self.config.target.min_target_spacing for saved_center in centers):
                continue
            centers.append(center)
            if len(centers) >= self.config.target.max_targets_per_contour:
                break
        return centers

    @staticmethod
    def _contour_center(contour: np.ndarray) -> Point:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            x, y, width, height = cv2.boundingRect(contour)
            return (x + width // 2, y + height // 2)
        return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    def _verify_target(self, mask: np.ndarray, center_x: int, center_y: int) -> float:
        radius = self.config.target.verify_radius
        top = max(center_y - radius, 0)
        bottom = min(center_y + radius + 1, mask.shape[0])
        left = max(center_x - radius, 0)
        right = min(center_x + radius + 1, mask.shape[1])

        if top >= bottom or left >= right:
            return 0.0

        patch = mask[top:bottom, left:right]
        if patch.size == 0:
            return 0.0

        yy, xx = np.ogrid[top:bottom, left:right]
        circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius * radius
        if not np.any(circle):
            return 0.0

        return float(np.count_nonzero(patch & circle) / np.count_nonzero(circle))

    def _deduplicate_targets(self, targets: Sequence[Target]) -> List[Target]:
        deduped: List[Target] = []
        min_spacing = self.config.target.min_target_spacing
        for target in targets:
            if any(self._distance(target.center, saved.center) < min_spacing for saved in deduped):
                continue
            deduped.append(target)
        return deduped

    @staticmethod
    def _boxes_intersect(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
        return not (first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1])

    @staticmethod
    def _merge_boxes(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)
