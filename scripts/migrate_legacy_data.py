"""Copy legacy local data to storage without changing source files or public URLs."""

import argparse
import json
import shutil
from pathlib import Path

from app.core.config import get_settings


def migrate(source: Path, destination: Path) -> int:
    source = source.resolve()
    destination = destination.resolve()
    files = []
    for name, expected in (("users.json", dict), ("histories.json", list)):
        path = source / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, expected):
                raise ValueError(f"{name} không đúng cấu trúc JSON.")
            files.append((path, destination / name))
    images = source / "image-cafe-input"
    if images.is_dir():
        files.extend((path, destination / "uploads" / path.name) for path in images.iterdir() if path.is_file())
    if not files:
        raise ValueError("Không tìm thấy dữ liệu cũ tại thư mục nguồn.")
    # Validate every destination before copying any data.
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Đích đã có dữ liệu; chọn thư mục đích trống: {destination}")
    if destination == source or destination in source.parents:
        raise ValueError("Thư mục đích không được trùng hoặc chứa thư mục nguồn.")
    for path, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream, path.open("rb") as original:
            shutil.copyfileobj(original, stream)
    return len(files)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=settings.project_dir)
    parser.add_argument("--destination", type=Path, default=settings.storage_dir)
    args = parser.parse_args()
    try:
        count = migrate(args.source, args.destination)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Không chuyển được dữ liệu: {error}\n")
    print(f"Đã sao chép {count} file. Dữ liệu nguồn được giữ nguyên.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
