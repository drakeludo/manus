import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orange_bot.bootstrap import TemplateBootstrapper


def main() -> None:
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    orange_paths, start_path = TemplateBootstrapper().build_from_directory(image_dir)
    print(f"orange_templates={','.join(str(path) for path in orange_paths)}")
    print(f"start_template={start_path}")


if __name__ == "__main__":
    main()
