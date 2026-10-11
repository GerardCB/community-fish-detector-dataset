"""
MIT Sea Grant River Herring Dataset
Source: https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip
Split logic: By location+video (train_test_split on location_video identifiers)
Category mapping:
  fish: river_herring, trout, perch, bass, eel, lamprey, sunfish, pickerel
  non-fish: none
  discard: unknown_fish, together with every image that contains one. These boxes
    are a mix of real fish and lamp glare, floating debris, bubbles and the
    timestamp overlay; discarding only the boxes would leave real fish unlabelled.
Images without annotations (~77% of frames) are kept.
"""

import json
import shutil
from pathlib import Path

import tqdm
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "mit_river_herring"
DATA_URL = "https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip"
CATEGORIES_FILTER = {
    "river_herring": "fish",
    "trout": "fish",
    "perch": "fish",
    "bass": "fish",
    "eel": "fish",
    "lamprey": "fish",
    "sunfish": "fish",
    "pickerel": "fish",
}
AMBIGUOUS_CATEGORIES = ["unknown_fish"]

settings = Settings()


def download_data(data_dir: Path):
    annotations_path = data_dir / "mit_river_herring" / "mit_sea_grant_river_herring.json"
    # main() creates data_dir before calling this, so check for the extracted files
    if annotations_path.exists():
        print(f"Data already exists: {data_dir}")
        return

    data_dir.mkdir(exist_ok=True, parents=True)

    print("Extracting data...")
    download_and_extract(data_dir, DATA_URL, DATASET_SHORTNAME)


def drop_images_with_ambiguous_annotations(annotations_path: Path, output_path: Path):
    """
    Drops images containing any AMBIGUOUS_CATEGORIES annotation, together with all
    their annotations. Discarding only the annotations would leave real fish
    unlabelled in the image.
    """
    if output_path.exists():
        print(f"Filtered annotation file already exists at {output_path}")
        return output_path

    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    ambiguous_category_ids = {
        c["id"] for c in coco_data["categories"] if c["name"] in AMBIGUOUS_CATEGORIES
    }
    ambiguous_image_ids = {
        a["image_id"]
        for a in coco_data["annotations"]
        if a["category_id"] in ambiguous_category_ids
    }
    coco_data["images"] = [
        i for i in coco_data["images"] if i["id"] not in ambiguous_image_ids
    ]
    coco_data["annotations"] = [
        a for a in coco_data["annotations"] if a["image_id"] not in ambiguous_image_ids
    ]
    print(
        f"Dropped {len(ambiguous_image_ids)} images with {AMBIGUOUS_CATEGORIES} annotations"
    )

    with open(output_path, "w") as f:
        json.dump(coco_data, f)

    return output_path


def get_location_and_video(image_filename: str) -> str:
    return "_".join(
        remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).split("_")[:2]
    )


def get_list_of_deployments_to_include_in_train_set(annotations_path: Path) -> list[str]:
    """Extract location and video names from the image filenames."""
    with open(annotations_path, "r") as f:
        images = json.load(f)["images"]
    # Sorted so the split does not depend on set iteration order
    all_deployments = sorted(
        {get_location_and_video(image["file_name"]) for image in images}
    )
    train_deployments, _ = train_test_split(
        all_deployments,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_deployments


def aggregate_images_in_one_folder(
    annotations_path: Path,
    images_path: Path,
    aggregated_annotations_path: Path,
    aggregated_images_path: Path,
):
    """
    The dataset is a collection of multiple coco datasets, one for each
    location and video. We join all into one single coco dataset.
    """
    if aggregated_images_path.exists():
        print(f"Aggregated images already exist: {aggregated_images_path}")
        return

    if aggregated_annotations_path.exists():
        print(f"Aggregated annotations already exist: {aggregated_annotations_path}")
        return

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    aggregated_images_path.mkdir(parents=True, exist_ok=True)

    for image in tqdm.tqdm(annotations["images"], total=len(annotations["images"])):
        original_image_path = images_path / image["file_name"]
        new_image_file_name = "_".join(Path(image["file_name"]).parts)
        new_image_path = aggregated_images_path / new_image_file_name
        shutil.copy(original_image_path, new_image_path)
        image["file_name"] = new_image_file_name

    with open(aggregated_annotations_path, "w") as f:
        json.dump(annotations, f)


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    annotations_path = raw_download_path / "mit_river_herring" / "mit_sea_grant_river_herring.json"
    images_path = raw_download_path / "mit_river_herring"

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    aggregated_annotations_path = processing_dir / settings.coco_file_name
    aggregated_images_path = processing_dir / settings.images_folder_name
    aggregate_images_in_one_folder(
        annotations_path,
        images_path,
        aggregated_annotations_path,
        aggregated_images_path,
    )

    filtered_annotations_path = processing_dir / "filtered_annotations.json"
    drop_images_with_ambiguous_annotations(
        aggregated_annotations_path, filtered_annotations_path
    )

    mapped_annotations_path = processing_dir / "mapped_annotations.json"
    map_annotations_to_fish_and_non_fish(
        filtered_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=aggregated_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(aggregated_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_deployments = set(
        get_list_of_deployments_to_include_in_train_set(mapped_annotations_path)
    )
    should_the_image_be_included_in_train_set = (
        lambda image_name: get_location_and_video(image_name) in train_deployments
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
        aggregated_images_path,
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
