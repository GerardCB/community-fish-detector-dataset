"""
Orange Chromide (Etroplus maculatus) — Annotated underwater fish detection dataset from pond environments
Source: https://data.mendeley.com/datasets/7w45jx35hd/1
Split logic: By blocks of 100 frame numbers (images are frames of pond videos, often only
    1-2 frames apart; the video is not in the filename and frame numbers restart per video,
    so frames sharing a number block go to the same split)
Category mapping:
  fish: the single YOLO class (every fish is boxed: orange chromides, other cichlids, koi)
  non-fish: none
  discard: none
"""

import json
import re
import shutil
import urllib.request
from pathlib import Path

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    CompressionType,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "orange_chromide"
DATA_URL = "https://data.mendeley.com/public-api/zip/7w45jx35hd/download/1"
# The YOLO labels have a single class (0) with no name; it is written to COCO as "fish"
CATEGORIES_FILTER = {
    "fish": "fish",
}

settings = Settings()

RAW_DATA_SUBDIR = (
    "Annotated underwater fish detection dataset from pond environments"
    "/Etroplus_maculatus"
)
IMAGE_SIZE = 640  # All images are 640x640
FRAME_BLOCK_SIZE = 100  # Frames whose numbers fall in the same block go to the same split

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / RAW_DATA_SUBDIR
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "orange_chromide_mapped_annotations.json"


def download_data():
    if raw_data_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / f"{DATASET_SHORTNAME}.zip"

    # Mendeley is behind Cloudflare, which answers `requests` (download_file) with a 403
    # challenge page; urllib is let through
    print(f"Downloading {DATA_URL} to {zip_path}")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f)

    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def yolo_to_coco(raw_data_dir: Path, coco_images_path: Path, coco_annotations_path: Path):
    """Convert YOLO annotations from all splits into a single COCO dataset."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    images = {}
    annotations = []
    image_id = 1
    ann_id = 1

    for split in ["Train", "Valid", "Test"]:
        images_dir = raw_data_dir / split / "Images"
        labels_dir = raw_data_dir / split / "Labels"

        if not images_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name

            if not image_path.exists():
                continue

            shutil.copy2(image_path, coco_images_path / image_name)

            images[image_name] = {
                "id": image_id,
                "file_name": image_name,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            }

            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    # YOLO format: class_id cx cy w h (normalized)
                    _, cx, cy, w, h = map(float, parts)

                    # Convert to COCO pixel coords [x, y, width, height]
                    bbox_w = w * IMAGE_SIZE
                    bbox_h = h * IMAGE_SIZE
                    bbox_x = (cx * IMAGE_SIZE) - (bbox_w / 2)
                    bbox_y = (cy * IMAGE_SIZE) - (bbox_h / 2)

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                        "area": round(bbox_w * bbox_h, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1

            image_id += 1

    coco_dict = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}],
    }

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def get_frame_block(image_name: str) -> int:
    """orange_chromide_frame1081_jpg.rf.<hash>.jpg -> 10"""
    return int(re.search(r"frame(\d+)_", image_name).group(1)) // FRAME_BLOCK_SIZE


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    yolo_to_coco(raw_data_dir, coco_images_path, coco_annotations_path)

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    # Split by frame-number block, reproducible: sorted blocks + fixed random_state
    with open(mapped_annotations_path) as f:
        image_names = [image["file_name"] for image in json.load(f)["images"]]
    train_blocks, _ = train_test_split(
        sorted({get_frame_block(name) for name in image_names}),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_blocks = set(train_blocks)

    should_the_image_be_included_in_train_set = (
        lambda image_path: get_frame_block(Path(image_path).name) in train_blocks
    )

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
