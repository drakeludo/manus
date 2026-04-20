import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orange_bot.click_model import OrangeClickNet


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model: OrangeClickNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["center_heatmap"], outputs["orange_mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export click model checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="models/orange_click.pt", help="Path to .pt checkpoint.")
    parser.add_argument("--output", default="models/orange_click.onnx", help="Output ONNX path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OrangeClickNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_width = int(checkpoint.get("image_width", 960))
    image_height = int(checkpoint.get("image_height", 544))
    dummy = torch.randn(1, 3, image_height, image_width, dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        OnnxWrapper(model),
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["center_heatmap", "orange_mask"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "image": {0: "batch"},
            "center_heatmap": {0: "batch"},
            "orange_mask": {0: "batch"},
        },
    )
    print(f"exported={output_path}")


if __name__ == "__main__":
    main()
