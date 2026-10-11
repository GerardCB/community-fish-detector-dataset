"""
TORSI (Trawl Observer Recording System Images) Dataset
Source: https://data.csiro.au/collection/64913
Split logic: By deployment. The 6 deployments (OP12, OP16, OP20, OP23, OP24, OP32) are
    separate sessions in the image timestamps: the 4 deployments of 2019-07-13/14/15 go
    to train, the 2 deployments of 2019-07-16/17 to val. Split ratio ~0.19
Category mapping:
  fish: orange_roughy, orange_roughy_edge, misc_fish, eel, oreo, whiptail, chimera, shark
  non-fish: sea_star, sea_urchin, brittle_star
  discard: mollusc (blurry, unidentifiable blobs), cnidaria (mostly sessile or
    unidentifiable), sea_feather (sea pens / stalked crinoids, sessile), sea_anemone,
    coral, siphonophore

Downloaded through the CSIRO Data Access Portal API: instances.json and the annotated
port-camera images only (the starboard stereo images have no annotations).
"""

import json
from pathlib import Path

import requests
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    split_coco_dataset_into_train_validation,
    map_annotations_to_fish_and_non_fish,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "torsi"
COLLECTION_FILES_URL = "https://data.csiro.au/dap/ws/v2/collections/64913/data"
CATEGORIES_FILTER = {
    "orange_roughy": "fish",
    "orange_roughy_edge": "fish",
    "misc_fish": "fish",
    "eel": "fish",
    "oreo": "fish",
    "whiptail": "fish",
    "chimera": "fish",
    "shark": "fish",
    "sea_star": "non-fish",
    "sea_urchin": "non-fish",
    "brittle_star": "non-fish",
    # discarded: mollusc, cnidaria, sea_feather, sea_anemone, coral, siphonophore
}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME / "data"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = raw_data_dir / "instances.json"
mapped_annotations_path = processing_dir / "torsi_mapped_annotations.json"
images_path = processing_dir / "JPEGImages"


def download_data():
    """
    Download instances.json and the port-camera images from the CSIRO Data Access
    Portal API. Files that are already present with the expected size are skipped.
    """
    response = requests.get(
        COLLECTION_FILES_URL, headers={"Accept": "application/json"}, timeout=120
    )
    response.raise_for_status()
    files = [
        f
        for f in response.json()["file"]
        if f["filename"] == "instances.json" or f["filename"].startswith("images/port/")
    ]

    to_download = []
    for f in files:
        target = raw_data_dir / f["filename"]
        if not (target.exists() and target.stat().st_size == f["fileSize"]):
            to_download.append((f, target))

    if not to_download:
        print(f"Data already downloaded: {raw_data_dir} ({len(files)} files)")
        return

    total_size = sum(f["fileSize"] for f, _ in to_download)
    print(f"Downloading {len(to_download)} of {len(files)} files ({total_size / 1e9:.1f} GB)")
    with requests.Session() as session:
        for f, target in tqdm(to_download):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with session.get(f["presignedLink"]["href"], stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(partial, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
            assert partial.stat().st_size == f["fileSize"], f"Incomplete download: {target}"
            partial.rename(target)


def adjust_path():
    """Remove relative path from json leaving only the name of the images."""
    with open(mapped_annotations_path, 'r', encoding='utf-8') as annotations_file:
        annotations_json = json.load(annotations_file)
    for image in annotations_json["images"]:
        image["file_name"] = Path(image["file_name"]).name

    with open(mapped_annotations_path, 'w', encoding='utf-8') as annotations_file:
        json.dump(annotations_json, annotations_file, indent=2)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    copy_images_to_processing(DATASET_SHORTNAME, raw_data_dir / "images" / "port")

    map_annotations_to_fish_and_non_fish(
        annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    adjust_path()

    add_dataset_shortname_prefix_to_image_names(
        images_path, mapped_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Split by date, which keeps each deployment whole: 4 deployments train, 2 val
    train_set_image_prefix = ["torsi_20190713", "torsi_20190714", "torsi_20190715"]

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).stem.split("-")[0] in train_set_image_prefix
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
        images_path,
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
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
