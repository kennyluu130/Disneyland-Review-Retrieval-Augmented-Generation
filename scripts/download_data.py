"""
Downloads the Disneyland Reviews dataset from Kaggle via kagglehub and copies
the CSV into data/raw/, matching the path expected by config.settings.raw_data_path.

Requires Kaggle credentials to be configured (~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY env vars) - see https://www.kaggle.com/docs/api
"""
import os
import shutil
import sys
from pathlib import Path

import kagglehub

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402


def main() -> None:
    path = kagglehub.dataset_download("arushchillar/disneyland-reviews")
    print(f"Downloaded dataset to {path}")

    src_csv = os.path.join(path, "DisneylandReviews.csv")
    dest_csv = settings.raw_data_path
    dest_csv.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_csv, dest_csv)
    print(f"Copied dataset to {dest_csv}")


if __name__ == "__main__":
    main()
