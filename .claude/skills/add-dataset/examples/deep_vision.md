# Example: deep_vision (URL download + CSV-to-COCO conversion + deployment-based split)

This is a representative example of a dataset script that:
- Downloads from a direct URL (zip file)
- Converts CSV annotations to COCO format
- Maps every source class to fish / non-fish
- Adds the dataset shortname prefix to image names
- Splits by deployment (trawl station) identifier extracted from filenames

```python
"""
Deep Vision Dataset
Source: https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip
Split logic: By trawl station (train_test_split on the station token of the filename,
    e.g. "ST019" in "ST019-1-20170511204109920.jpg" or "ST7" in "ST7_129-...jpg")
Category mapping:
  fish: herring, mackerel, bluewhiting, mesopelagic (lanternfish-type fish),
    Sharpchin barracudina, spotted barracudina, salmon, lumpsucker
  non-fish: none
  discard: none

Annotations come from several CSVs that partly overlap: a few images appear in more
than one CSV, annotated separately. Each image takes its boxes from the first CSV
that lists it, so its fish are not boxed twice. Some CSV rows give the wrong
species folder for an image (e.g. "test/mackerel/" for a file in "test/mixed/"); those
images are looked up by filename instead.
"""

import csv
import json
import re
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    CompressionType,
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "deep_vision"
DATA_URL = (
    "https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip"
)
CATEGORIES_FILTER = {
    "herring": "fish",
    "mackerel": "fish",
    "bluewhiting": "fish",
    "mesopelagic": "fish",
    "Sharpchin barracudina": "fish",
    "spotted barracudina": "fish",
    "salmon": "fish",
    "lumpsucker": "fish",
}

settings = Settings()


def download_data(download_path: Path):
    if (download_path / "fish_dataset").exists():
        print(f"Data already exists: {download_path}")
        return
    download_and_extract(
        download_path, DATA_URL, DATASET_SHORTNAME, CompressionType.ZIP
    )


def csvs_to_coco(download_dir: Path, csv_files, images_path, output_json):
    """
    Converts multiple CSV files with annotations to a COCO-format JSON file.
    Each image takes its annotations from the first CSV that lists it.
    """
    images = {}
    image_source_csv = {}
    # Some CSV rows point to the wrong species folder, so fall back to the filename
    image_name_to_path = {
        image_path.name: image_path
        for image_path in sorted((download_dir / "fish_dataset").glob("20*/test/*/*.jpg"))
        + sorted((download_dir / "fish_dataset").glob("20*/train/source-images/*/*.jpg"))
    }
    annotations = []
    categories = {}

    ann_id = 1
    image_id = 1

    for csv_file in csv_files:
        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue
                rel_file_name, xmin, ymin, xmax, ymax, label = row

                rel_file_name = rel_file_name.lstrip("/")
                raw_image_path = download_dir / "fish_dataset" / rel_file_name
                file_name = raw_image_path.name
                if not raw_image_path.exists():
                    raw_image_path = image_name_to_path.get(file_name)
                    if raw_image_path is None:
                        print(f"Image not found, skipping: {rel_file_name}")
                        continue

                # Skip images already annotated by an earlier CSV
                if image_source_csv.setdefault(file_name, csv_file) != csv_file:
                    continue

                try:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                except ValueError:
                    continue

                if file_name not in images:
                    shutil.copy2(raw_image_path, images_path / file_name)
                    with Image.open(raw_image_path) as img:
                        image_width, image_height = img.size
                    images[file_name] = {
                        "id": image_id,
                        "file_name": file_name,
                        "width": image_width,
                        "height": image_height,
                    }
                    image_id += 1

                width = xmax - xmin
                height = ymax - ymin

                if label not in categories:
                    cat_id = len(categories) + 1
                    categories[label] = {
                        "id": cat_id,
                        "name": label,
                        "supercategory": label,
                    }
                cat_id = categories[label]["id"]

                ann = {
                    "id": ann_id,
                    "image_id": images[file_name]["id"],
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
                annotations.append(ann)
                ann_id += 1

    coco_images = list(images.values())
    coco_categories = list(categories.values())

    coco_dict = {
        "images": coco_images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"COCO annotations saved to {output_json}")
    print(f"Number of images: {len(coco_images)}")
    print(f"Number of annotations: {len(annotations)}")


def create_coco_dataset(download_dir: Path, images_path: Path, annotations_path: Path):
    """Convert CSV annotations to COCO format."""
    # The per-year CSVs come first; the root-level test/val CSVs mostly repeat them
    # but also annotate a few images the per-year CSVs miss.
    csv_files = [
        download_dir / "fish_dataset/2017/train/source-train2017-annotations.csv",
        download_dir / "fish_dataset/2017/test/test_2017_annotations.csv",
        download_dir / "fish_dataset/2018/train/source-train2018-annotations.csv",
        download_dir / "fish_dataset/2018/test/test_2018_annotations.csv",
        download_dir / "fish_dataset/test_annotations.csv",
        download_dir / "fish_dataset/val_annotations.csv",
    ]

    if annotations_path.exists():
        print(f"COCO dataset already exists: {annotations_path}")
    else:
        csvs_to_coco(download_dir, csv_files, images_path, annotations_path)

    return images_path, annotations_path


def get_station(image_filename: str) -> str:
    """
    Station token of the filename: "ST019" for "ST019-1-20170511204109920.jpg" (2017)
    and "ST7" for "ST7_129-20180507063944924.jpg" (2018).
    """
    image_filename = remove_dataset_shortname_prefix_from_image_filename(
        Path(image_filename).name, DATASET_SHORTNAME
    )
    return re.split(r"[-_]", image_filename)[0]


def get_list_of_stations_to_include_in_train_set(annotations_path: Path) -> list[str]:
    with open(annotations_path, "r") as f:
        images = json.load(f)["images"]
    # Sorted so the split does not depend on set iteration order
    stations = sorted({get_station(image["file_name"]) for image in images})
    train_stations, _ = train_test_split(
        stations,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_stations


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    create_coco_dataset(raw_download_path, coco_images_path, coco_annotations_path)

    mapped_annotations_path = processing_dir / "mapped_annotations.json"
    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path, mapped_annotations_path, DATASET_SHORTNAME
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_stations = set(
        get_list_of_stations_to_include_in_train_set(mapped_annotations_path)
    )
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_station(image_path) in train_stations
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


if __name__ == "__main__":
    main()
```
