from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisionRegion:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


@dataclass(frozen=True)
class ScreenConfig:
    use_full_screen: bool = True
    tree_search_expansion: int = 120
    max_tree_candidates: int = 2


@dataclass(frozen=True)
class OrangeColorThreshold:
    min_red: int = 132
    min_rg_gap: int = 3
    max_blue: int = 205
    min_brightness: int = 18
    hue_min: int = 2
    hue_max: int = 42
    min_saturation: int = 45
    min_value: int = 28
    shadow_hue_min: int = 0
    shadow_hue_max: int = 30
    shadow_min_saturation: int = 18
    shadow_min_value: int = 10


@dataclass(frozen=True)
class TargetLimits:
    min_pixels: int = 20
    max_pixels: int = 12000
    min_fill_ratio: float = 0.10
    min_circularity: float = 0.12
    max_aspect_ratio: float = 3.5
    min_target_spacing: int = 24
    verify_radius: int = 8
    min_verify_ratio: float = 0.06
    peak_threshold_ratio: float = 0.48
    min_peak_distance: float = 3.0
    max_targets_per_contour: int = 40
    close_kernel_size: int = 7
    open_kernel_size: int = 3
    dilate_kernel_size: int = 7
    dilate_iterations: int = 2
    grid_step: int = 14
    min_component_width: int = 8
    min_component_height: int = 8


@dataclass(frozen=True)
class VisionConfig:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    region: VisionRegion = field(default_factory=VisionRegion)
    orange_rgb: OrangeColorThreshold = field(default_factory=OrangeColorThreshold)
    target: TargetLimits = field(default_factory=TargetLimits)


@dataclass(frozen=True)
class CustomDetectorConfig:
    name: str
    model_path: str
    enabled: bool = True
    input_width: int = 960
    input_height: int = 544
    confidence_threshold: float = 0.45
    mask_threshold: float = 0.40
    mask_weight: float = 0.35
    min_peak_distance: int = 18
    peak_kernel_size: int = 5
    max_detections: int = 96
    heatmap_output_name: str = "center_heatmap"
    mask_output_name: str = "orange_mask"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)


@dataclass(frozen=True)
class YoloModelConfig:
    name: str
    model_path: str
    enabled: bool = True
    confidence_threshold: float = 0.22
    image_size: int = 640
    target_class_id: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class TemplateConfig:
    orange_template_glob: str = "models/orange*.png"
    orange_template_path: str = "models/orange.png"
    start_template_path: str = "models/minigame_start.png"
    orange_threshold: float = 0.66
    start_threshold: float = 0.72
    orange_scales: tuple[float, ...] = (0.75, 0.90, 1.0, 1.10, 1.25)
    start_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    dedupe_distance: int = 28
    enabled: bool = True


@dataclass(frozen=True)
class StateConfig:
    required_active_frames: int = 1
    required_inactive_frames: int = 3
    min_targets_for_active: int = 1
    yolo_presence_threshold: int = 1
    color_presence_threshold: int = 4
    template_presence_threshold: int = 1


@dataclass(frozen=True)
class BrainConfig:
    cluster_distance: int = 26
    max_targets: int = 24
    min_click_score: float = 0.08
    primary_weight: float = 1.2
    secondary_weight: float = 1.15
    color_weight: float = 1.35
    template_weight: float = 0.90
    dual_model_bonus: float = 0.65
    color_confirmation_bonus: float = 0.45
    template_confirmation_bonus: float = 0.25
    require_yolo_support: bool = False
    allow_color_only_fallback: bool = True


@dataclass(frozen=True)
class ControlConfig:
    forward_key: str = "w"
    back_key: str = "s"
    interact_key: str = "e"
    trigger_back_duration: float = 0.65
    trigger_forward_duration: float = 0.85
    trigger_settle_delay: float = 0.04
    interact_wait: float = 0.25
    click_cooldown: float = 0.0
    max_clicks_per_tree: int = 96
    max_cycle_seconds: float = 10.0
    retry_delay: float = 0.01
    post_batch_delay: float = 0.0
    post_rearm_delay: float = 0.20
    toggle_debounce: float = 0.5
    loop_sleep: float = 0.1
    click_position_ttl: float = 0.25
    click_position_radius: int = 8
    mouse_settle_delay: float = 0.0
    mouse_press_delay: float = 0.0
    scan_burst_count: int = 7
    scan_burst_delay: float = 0.0
    empty_scan_streak_to_rearm: int = 2
    click_spread_enabled: bool = True
    click_spread_radius: int = 26
    click_effect_radius: int = 26
    coverage_fill_enabled: bool = True
    dataset_path: str = "datasets"
    screenshot_key: str = "f8"
    click_spread_pattern: tuple[tuple[int, int], ...] = (
        (0, 0),
        (-12, 0),
        (12, 0),
        (0, -12),
        (0, 12),
        (-18, -18),
        (18, -18),
        (-18, 18),
        (18, 18),
    )


@dataclass(frozen=True)
class BotConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    controls: ControlConfig = field(default_factory=ControlConfig)
    custom_primary: CustomDetectorConfig = field(
        default_factory=lambda: CustomDetectorConfig(
            name="custom_primary",
            model_path="models/orange_click.onnx",
            enabled=True,
        )
    )
    custom_secondary: CustomDetectorConfig = field(
        default_factory=lambda: CustomDetectorConfig(
            name="custom_secondary",
            model_path="models/orange_click_refine.onnx",
            enabled=True,
            confidence_threshold=0.16,
            mask_threshold=0.14,
        )
    )
    yolo_primary: YoloModelConfig = field(
        default_factory=lambda: YoloModelConfig(
            name="primary",
            model_path="models/orange_primary.pt",
            enabled=False,
        )
    )
    yolo_secondary: YoloModelConfig = field(
        default_factory=lambda: YoloModelConfig(
            name="secondary",
            model_path="models/orange_secondary.pt",
            enabled=False,
        )
    )
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    state: StateConfig = field(default_factory=StateConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)


DEFAULT_CONFIG = BotConfig()
