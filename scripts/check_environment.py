"""Validate settings, installed requirements and private models before running."""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from app.core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true", help="Load checkpoints and validate class knowledge")
    args = parser.parse_args()
    try:
        settings = get_settings()
        if sys.version_info[:2] != (3, 13):
            raise ValueError("Bản bàn giao được kiểm chứng với Python 3.13.")
        for line in (settings.project_dir / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if "==" not in line or line.startswith("#"):
                continue
            package, expected = line.split("==", 1)
            installed = version(package.split("[")[0])
            if installed.split("+")[0] != expected:
                raise ValueError(f"{package}: cần {expected}, đang cài {installed}")
        settings.validate_models()
        from app.repositories.knowledge import get_disease_knowledge

        get_disease_knowledge()
        print(f"Python {sys.version.split()[0]}; device={settings.device}")
        print(f"Models: {settings.model_dir}")
        print(f"Storage: {settings.storage_dir}")
        if args.load_models:
            from app.services.predictor_service import get_pipeline

            pipeline = get_pipeline()
            print(f"Đã nạp pipeline trên {pipeline.device}; {len(pipeline.class_names)} lớp.")
        print("Kiểm tra môi trường thành công. Chạy thêm: python -m pip check")
        return 0
    except (ValueError, FileNotFoundError, PackageNotFoundError, ImportError) as error:
        print(f"Kiểm tra thất bại: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
