import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orange_bot.click_model import ClickPoint, OrangeClickNet, build_center_heatmap, build_click_mask, extract_click_points


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom click model (heatmap + mask).")
    parser.add_argument("--data-dir", default="datasets/click_oranges", help="Dataset root.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--image-width", type=int, default=960, help="Training width.")
    parser.add_argument("--image-height", type=int, default=544, help="Training height.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-model", default="models/orange_click.pt", help="Best checkpoint output path.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_image(image_path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


class ClickDataset(Dataset):
    def __init__(self, root: Path, split: str, image_width: int, image_height: int, augment: bool):
        self.root = root
        self.split = split
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        self.samples = [
            Sample(image_path=image_path, label_path=label_dir / f"{image_path.stem}.json")
            for image_path in sorted(image_dir.iterdir())
            if image_path.is_file() and (label_dir / f"{image_path.stem}.json").exists()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_bgr = read_image(sample.image_path)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_rgb.shape[:2]
        label_data = json.loads(sample.label_path.read_text(encoding="utf-8"))
        points = [
            ClickPoint(
                x=int(item["x"]),
                y=int(item["y"]),
                radius=int(item.get("radius", 16)),
            )
            for item in label_data.get("points", [])
        ]

        resized = cv2.resize(image_rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        scale_x = self.image_width / max(1, image_width)
        scale_y = self.image_height / max(1, image_height)
        scaled_points = [
            ClickPoint(
                x=int(round(point.x * scale_x)),
                y=int(round(point.y * scale_y)),
                radius=max(4, int(round(point.radius * (scale_x + scale_y) * 0.5))),
            )
            for point in points
        ]

        if self.augment:
            resized, scaled_points = self._augment(resized, scaled_points)

        heatmap = build_center_heatmap(self.image_height, self.image_width, scaled_points)
        mask = build_click_mask(self.image_height, self.image_width, scaled_points)

        image_tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1)))
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "mask": mask_tensor,
            "point_count": torch.tensor(len(scaled_points), dtype=torch.int64),
        }

    def _augment(self, image: np.ndarray, points: list[ClickPoint]) -> tuple[np.ndarray, list[ClickPoint]]:
        image = image.copy()
        points = list(points)

        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            points = [ClickPoint(x=self.image_width - 1 - point.x, y=point.y, radius=point.radius) for point in points]

        if random.random() < 0.35:
            alpha = random.uniform(0.85, 1.20)
            beta = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.20:
            image = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

        return image, points


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - score.mean()


def evaluate(model: OrangeClickNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    matched = 0
    total_expected = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            heatmap_targets = batch["heatmap"].to(device)
            mask_targets = batch["mask"].to(device)
            outputs = model(images)
            heatmap_logits = outputs["center_heatmap"]
            mask_logits = outputs["orange_mask"]
            loss = (
                criterion(heatmap_logits, heatmap_targets)
                + criterion(mask_logits, mask_targets)
                + 0.5 * dice_loss(mask_logits, mask_targets)
            )
            total_loss += float(loss.item()) * images.size(0)

            heatmaps = torch.sigmoid(heatmap_logits).cpu().numpy()
            masks = torch.sigmoid(mask_logits).cpu().numpy()
            target_maps = heatmap_targets.cpu().numpy()
            for heatmap, mask, target_map in zip(heatmaps, masks, target_maps):
                predicted_points = extract_click_points(heatmap[0], mask[0], confidence_threshold=0.18, mask_threshold=0.18)
                expected_ys, expected_xs = np.where(target_map[0] >= 0.50)
                expected_points = list(zip(expected_xs.tolist(), expected_ys.tolist()))
                total_expected += len(expected_points)
                remaining = expected_points[:]
                for px, py, _ in predicted_points:
                    best_index = None
                    best_distance = None
                    for index, (ex, ey) in enumerate(remaining):
                        distance = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
                        if distance <= 18 and (best_distance is None or distance < best_distance):
                            best_distance = distance
                            best_index = index
                    if best_index is not None:
                        matched += 1
                        remaining.pop(best_index)

    dataset_size = max(1, len(loader.dataset))
    recall = matched / max(1, total_expected)
    return total_loss / dataset_size, recall


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    train_dataset = ClickDataset(data_dir, "train", args.image_width, args.image_height, augment=True)
    val_dataset = ClickDataset(data_dir, "val", args.image_width, args.image_height, augment=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("Dataset is empty. Run export_click_dataset.py first.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = OrangeClickNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_recall = -1.0
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            heatmap_targets = batch["heatmap"].to(device)
            mask_targets = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            heatmap_logits = outputs["center_heatmap"]
            mask_logits = outputs["orange_mask"]
            loss = (
                criterion(heatmap_logits, heatmap_targets)
                + criterion(mask_logits, mask_targets)
                + 0.5 * dice_loss(mask_logits, mask_targets)
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / max(1, len(train_dataset))
        val_loss, val_recall = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_center_recall={val_recall:.4f}"
        )

        if val_recall >= best_recall:
            best_recall = val_recall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "best_recall": best_recall,
                },
                output_path,
            )
            print(f"saved_best={output_path}")


if __name__ == "__main__":
    main()
