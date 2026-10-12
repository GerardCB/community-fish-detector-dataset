"""
KakaduFishAI Dataset
Source: https://zenodo.org/records/7250921 (202210-KakaduFishAI-TrainingData.zip)
Split logic: Random by image, reproducible (file names are shuffled ids, no
    deployment/video information is available)
Category mapping:
  fish: all 23 classes (freshwater fish species)
  non-fish: none
  discard: none

The zip is compressed with Deflate64, which Python's zipfile cannot decompress, so it is
extracted with the system `unzip`.
"""

import json
import shutil
import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "kakadu"
SOURCE_URL = "https://zenodo.org/records/7250921/files/202210-KakaduFishAI-TrainingData.zip?download=1"
CATEGORIES_FILTER = {
    "Ambassis agrammus": "fish",
    "Ambassis macleayi": "fish",
    "Amniataba percoides": "fish",
    "Craterocephalus sturcusmuscarum": "fish",
    "Denariusa bandata": "fish",
    "Glossamia aprion": "fish",
    "Glossogobius spp.": "fish",
    "Hephaestus fuliginosus": "fish",
    "Lates calcarifer": "fish",
    "Leiopotherapon unicolor": "fish",
    "Liza ordensis": "fish",
    "Megalops cyprinoides": "fish",
    "Melanotaenia nigrans": "fish",
    "Melanotaenia splendida inornata": "fish",
    "Mogurnda mogurnda": "fish",
    "Nematalosa erebi": "fish",
    "Neoarius spp.": "fish",
    "Neosilurus spp.": "fish",
    "Oxyeleotris lineolata": "fish",
    "Scleropages jardinii": "fish",
    "Strongylura krefftii": "fish",
    "Sycomistes butleri": "fish",
    "Toxotes chatareus": "fish",
}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
raw_annotations_path = raw_data_dir / "KakaduFishAI_boundingbox.json"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
mapped_annotations_path = processing_dir / "kakadu_mapped_annotations.json"
images_path = processing_dir / settings.images_folder_name


def download_data():
    """Download the training data zip from Zenodo and extract it with `unzip`."""
    if raw_annotations_path.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_dir / f"{DATASET_SHORTNAME}.zip"
    if not zip_path.exists():
        partial = zip_path.with_name(zip_path.name + ".part")
        download_file(SOURCE_URL, partial)
        partial.rename(zip_path)

    # Deflate64-compressed: Python's zipfile (extract_downloaded_file) cannot extract it
    print(f"Extracting {zip_path} to {raw_data_dir}")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(raw_data_dir)], check=True)
    zip_path.unlink()


def copy_images():
    images_path.mkdir(parents=True, exist_ok=True)
    with open(raw_annotations_path) as f:
        file_names = [image["file_name"] for image in json.load(f)["images"]]
    for file_name in tqdm(file_names):
        target = images_path / file_name
        if not target.exists() and not (images_path / f"{DATASET_SHORTNAME}_{file_name}").exists():
            shutil.copy2(raw_data_dir / file_name, target)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    copy_images()

    map_annotations_to_fish_and_non_fish(
        raw_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path, mapped_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Random split by image, reproducible: sorted file names + fixed random_state
    with open(mapped_annotations_path) as f:
        image_names = sorted(Path(image["file_name"]).name for image in json.load(f)["images"])
    train_images, _ = train_test_split(
        image_names,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_images = set(train_images)

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).name in train_images
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
