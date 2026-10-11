"""
Roboflow Fish Dataset
Source: https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6
Split logic: Pre-provided split (uses Roboflow's original train/val/test split)
Category mapping:
  fish: all 26 source classes (fish families with Latin names, e.g. "Scombridae -Tunas-",
    and the same families with common names, e.g. "tuna"), plus Roboflow's parent
    category "fish"
  non-fish: none
  discard: none

The export contains most images twice: pixel-identical copies with identical boxes,
one labelled with Latin family names and one with common names. Only one copy of each
source image is kept.
"""

import json
import shutil
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    extract_downloaded_file,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "roboflow_fish"
CATEGORIES_FILTER = {
    "fish": "fish",
    "Acanthuridae -Surgeonfishes-": "fish",
    "Balistidae -Triggerfishes-": "fish",
    "Carangidae -Jacks-": "fish",
    "Ephippidae -Spadefishes-": "fish",
    "Labridae -Wrasse-": "fish",
    "Lutjanidae -Snappers-": "fish",
    "Pomacanthidae -Angelfishes-": "fish",
    "Pomacentridae -Damselfishes-": "fish",
    "Scaridae -Parrotfishes-": "fish",
    "Scombridae -Tunas-": "fish",
    "Serranidae -Groupers-": "fish",
    "Shark -Selachimorpha-": "fish",
    "Zanclidae (Moorish Idol)": "fish",
    "Zanclidae -Moorish Idol-": "fish",
    "angel": "fish",
    "damsel": "fish",
    "grouper": "fish",
    "jack": "fish",
    "parrot": "fish",
    "shark": "fish",
    "snapper": "fish",
    "spade": "fish",
    "surgeon": "fish",
    "trigger": "fish",
    "tuna": "fish",
    "wrasse": "fish",
}

settings = Settings()


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6"
    data_path = data_dir / "roboflow_fish.zip"

    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, data_path)
        print("Extracting data...")
        extract_downloaded_file(data_path, data_dir)


def join_all_images_and_annotations_into_single_coco_dataset(
    data_dir: Path, coco_images_dir: Path, coco_annotations_path: Path
):
    """
    Merges all train/val/test splits into a single COCO dataset, keeping one copy of
    each source image (Roboflow names images "<source>_<ext>.rf.<hash>.jpg").
    """
    splits = ["train", "valid", "test"]
    split_dirs = [data_dir / split for split in splits]

    for split_dir in split_dirs:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotation_file}")

    coco_images_dir.mkdir(exist_ok=True)

    combined_annotations = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0
    seen_source_images = set()
    duplicate_count = 0

    for split_dir in split_dirs:
        annotation_file = split_dir / "_annotations.coco.json"
        with open(annotation_file) as f:
            split_annotations = json.load(f)

        duplicate_image_ids = set()
        for image in split_annotations["images"]:
            source_image = image["file_name"].rsplit(".rf.", 1)[0]
            if source_image in seen_source_images:
                duplicate_image_ids.add(image["id"])
                continue
            seen_source_images.add(source_image)
            image["id"] += image_id_offset
            src_path = split_dir / image["file_name"]
            dst_path = coco_images_dir / image["file_name"]
            shutil.copy2(src_path, dst_path)
            combined_annotations["images"].append(image)
        duplicate_count += len(duplicate_image_ids)

        for annotation in split_annotations["annotations"]:
            if annotation["image_id"] in duplicate_image_ids:
                continue
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            combined_annotations["annotations"].append(annotation)

        if not combined_annotations["categories"]:
            combined_annotations["categories"] = split_annotations["categories"]

        image_id_offset = max(img["id"] for img in combined_annotations["images"]) + 1
        if combined_annotations["annotations"]:
            annotation_id_offset = (
                max(ann["id"] for ann in combined_annotations["annotations"]) + 1
            )

    with open(coco_annotations_path, "w") as f:
        json.dump(combined_annotations, f)

    print(f"Skipped {duplicate_count} duplicate copies of source images")
    print(f"Combined dataset saved to {coco_images_dir}")
    print(f"Total images: {len(combined_annotations['images'])}")
    print(f"Total annotations: {len(combined_annotations['annotations'])}")

    return coco_images_dir, coco_annotations_path


def get_list_of_cameras_to_include_in_train_set(train_image_folder: Path) -> list[str]:
    """Using same split as provided by Roboflow."""
    train_image_names = list(train_image_folder.glob("*.jpg"))
    train_image_names = [f"{DATASET_SHORTNAME}_{image_name.name}" for image_name in train_image_names]
    return train_image_names


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

    coco_images_path, coco_annotations_path = (
        join_all_images_and_annotations_into_single_coco_dataset(
            raw_download_path, coco_images_path, coco_annotations_path
        )
    )

    # Categories are mapped by name, so Roboflow's 0-indexed ids need no conversion
    mapped_annotations_path = processing_dir / "annotations_coco_mapped.json"
    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    raw_train_set = raw_download_path / "train"
    train_image_names = get_list_of_cameras_to_include_in_train_set(raw_train_set)
    should_the_image_be_included_in_train_set = (
        lambda image_path: image_path in train_image_names
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
