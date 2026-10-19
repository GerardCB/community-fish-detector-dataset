"""
PEBL Seaweed and Shellfish Farm Fauna — still frames from fixed underwater cameras
(PEBL SubCam) on two UK seaweed and shellfish farms (2020, 2024 and 2026 deployments)
Source: LILA BC (https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/pebl_farm_fauna_v1.0.zip)
Split logic: By camera location (the anonymised camera id in each image's "location" field):
    farm_A_sc14 and farm_A_bed1 → val; farm_A_sc11, farm_A_cam (2020) and farm_B_cam → train
Category mapping:
  fish: Fish, Scooter (dragonets and gobies), Flatfish, Elasmobranch
  non-fish: Hermit crab, Crab, Jellyfish (moon and lion's mane jellies), Starfish,
    Gastropod (whelk-like sea snails), Squid, Shrimp
  discard: none
  (Elasmobranch, Squid and Shrimp have no boxes in v1.0)
"""

import json
import shutil
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    CompressionType,
    download_file,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "pebl_farm_fauna"
DATA_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "community-fish-detection-dataset/pebl_farm_fauna_v1.0.zip"
)
CATEGORIES_FILTER = {
    "Fish": "fish",
    "Scooter": "fish",
    "Flatfish": "fish",
    "Elasmobranch": "fish",
    "Hermit crab": "non-fish",
    "Crab": "non-fish",
    "Jellyfish": "non-fish",
    "Starfish": "non-fish",
    "Gastropod": "non-fish",
    "Squid": "non-fish",
    "Shrimp": "non-fish",
}
VAL_LOCATIONS = {"farm_A_sc14", "farm_A_bed1"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "pebl_farm_fauna"
raw_images_dir = raw_data_dir / "images"
raw_annotations_path = raw_data_dir / "pebl_farm_fauna_coco.json"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "pebl_farm_fauna_mapped_annotations.json"


def download_data():
    if raw_annotations_path.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "pebl_farm_fauna_v1.0.zip"
    download_file(DATA_URL, zip_path)
    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def flatten_coco(coco_images_path: Path, coco_annotations_path: Path):
    """
    Copy the images into one folder and rewrite the COCO Camera Traps JSON with integer
    ids. Frame names repeat across videos (images/<farm>/<video>/frame_<n>.jpg), so each
    image is renamed <video>_frame_<n>.jpg. The location, seq_id, frame_num and datetime
    fields are kept on each image.
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    with open(raw_annotations_path) as f:
        coco_data = json.load(f)

    image_id_map = {}
    for new_id, image in enumerate(coco_data["images"], start=1):
        image_id_map[image["id"]] = new_id
        source_path = raw_images_dir / image["file_name"]
        new_file_name = f"{image['seq_id']}_{Path(image['file_name']).name}"
        shutil.copy2(source_path, coco_images_path / new_file_name)
        image["id"] = new_id
        image["file_name"] = new_file_name

    for annotation in coco_data["annotations"]:
        annotation["image_id"] = image_id_map[annotation["image_id"]]

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    flatten_coco(coco_images_path, coco_annotations_path)

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        val_image_names = {
            image["file_name"]
            for image in json.load(f)["images"]
            if image["location"] in VAL_LOCATIONS
        }

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).name not in val_image_names
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
