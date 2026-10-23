"""
AFSC/RACE/MACE 2019-2020 Salmon and Pollock Object Detection
Source: https://www.fisheries.noaa.gov/inport/item/69412
Data: public GCS bucket gs://nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/annotated_data/
Split logic: By tow (train_test_split on the 7 <year>_t<tow> identifiers)
Category mapping:
  fish: Pollock, Salmon, Fish (unidentified), Herring
  non-fish: none
  discard: Other, together with every frame that contains one. These boxes are
    blurry, unclear shapes, partly fish-like; discarding only the boxes would
    leave possible fish unlabelled.
    Shark, Invertebrate, Krill, Occlusion, No Fish, Fish Low, Fish Med,
    Fish High, Low Light, Low Visibility are listed as categories but have no
    boxes (most are frame-level CVAT tags).
Frames without boxes (~1,100, tagged "No Fish") are kept.
Frames are consecutive 30 fps video frames from 184 clips of ~60 frames each,
so neighbouring frames are highly redundant. All frames are kept.
"""

import json
import shutil
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import tqdm
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "afsc_salmon_pollock"
BUCKET_LIST_URL = "https://storage.googleapis.com/storage/v1/b/nmfs_odp_afsc/o"
BUCKET_DOWNLOAD_URL = "https://storage.googleapis.com/nmfs_odp_afsc"
DATA_PREFIX = "RACE/MACE/salmon_pollock_object_detection/annotated_data/"
CATEGORIES_FILTER = {
    "Pollock": "fish",
    "Salmon": "fish",
    "Fish": "fish",
    "Herring": "fish",
}
AMBIGUOUS_CATEGORIES = ["Other"]
DOWNLOAD_WORKERS = 16

settings = Settings()


def list_bucket_files() -> list[dict]:
    """Lists every object (name, size) under DATA_PREFIX in the public bucket."""
    items = []
    page_token = None
    while True:
        params = {
            "prefix": DATA_PREFIX,
            "fields": "items(name,size),nextPageToken",
            "maxResults": "1000",
        }
        if page_token:
            params["pageToken"] = page_token
        response = requests.get(BUCKET_LIST_URL, params=params)
        response.raise_for_status()
        page = response.json()
        items += page.get("items", [])
        page_token = page.get("nextPageToken")
        if not page_token:
            return items


def download_bucket_file(item: dict, data_dir: Path):
    save_path = data_dir / item["name"][len(DATA_PREFIX) :]
    if save_path.exists() and save_path.stat().st_size == int(item["size"]):
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_DOWNLOAD_URL}/{urllib.parse.quote(item['name'])}"
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
    # Write to a temporary file so an interrupted run never leaves a truncated image
    tmp_path = save_path.with_suffix(save_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    tmp_path.rename(save_path)


def download_data(data_dir: Path):
    """Downloads the 184 annotated clips (one COCO json + PNG frames each), ~29 GB."""
    done_marker = data_dir / ".download_complete"
    if done_marker.exists():
        print(f"Data already exists: {data_dir}")
        return

    items = [
        item
        for item in list_bucket_files()
        if not item["name"].endswith("/") and Path(item["name"]).name != ".DS_Store"
    ]
    print(f"Downloading {len(items)} files to {data_dir}...")
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        list(
            tqdm.tqdm(
                executor.map(lambda item: download_bucket_file(item, data_dir), items),
                total=len(items),
            )
        )
    done_marker.touch()


def aggregate_clips(
    data_dir: Path,
    aggregated_annotations_path: Path,
    aggregated_images_path: Path,
):
    """
    The dataset has one COCO file per clip, with category ids that differ between
    clips. We join all into one COCO dataset, matching categories by name, and
    copy the frames into one folder as <clip>_frame_<n>.PNG.
    """
    if aggregated_annotations_path.exists():
        print(f"Aggregated annotations already exist: {aggregated_annotations_path}")
        return

    clip_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    clip_annotations = {
        clip_dir.name: json.loads((clip_dir / f"{clip_dir.name}.json").read_text())
        for clip_dir in clip_dirs
    }

    category_names = sorted(
        {c["name"] for coco in clip_annotations.values() for c in coco["categories"]}
    )
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(category_names)]
    category_name_to_id = {c["name"]: c["id"] for c in categories}

    aggregated_images_path.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for clip, coco in tqdm.tqdm(clip_annotations.items()):
        clip_category_to_id = {
            c["id"]: category_name_to_id[c["name"]] for c in coco["categories"]
        }
        image_id_map = {}
        for image in coco["images"]:
            # file_name is a URL under .../data/<clip>/frames/, but the bucket folder is annotated_data/
            frame_name = Path(image["file_name"]).name
            new_file_name = f"{clip}_{frame_name}"
            new_image_path = aggregated_images_path / new_file_name
            if not new_image_path.exists():
                shutil.copy(data_dir / clip / "frames" / frame_name, new_image_path)

            image_id_map[image["id"]] = len(images) + 1
            images.append(
                {
                    "id": image_id_map[image["id"]],
                    "file_name": new_file_name,
                    "width": image["width"],
                    "height": image["height"],
                    "cvat_tags": image.get("cvat_tags", []),
                }
            )
        for annotation in coco["annotations"]:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id_map[annotation["image_id"]],
                    "category_id": clip_category_to_id[annotation["category_id"]],
                    "bbox": annotation["bbox"],
                    "area": annotation["bbox"][2] * annotation["bbox"][3],
                    "iscrowd": 0,
                    "attributes": annotation.get("attributes", {}),
                }
            )

    with open(aggregated_annotations_path, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories}, f
        )


def drop_images_with_ambiguous_annotations(annotations_path: Path, output_path: Path):
    """
    Drops images containing any AMBIGUOUS_CATEGORIES annotation, together with all
    their annotations. Discarding only the annotations would leave possible fish
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


def get_tow(image_filename: str) -> str:
    """<year>_t<tow>_vid<n>_<start>_<end>_frame_<n>.PNG -> <year>_t<tow>"""
    return "_".join(
        remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).split("_")[:2]
    )


def get_list_of_tows_to_include_in_train_set(annotations_path: Path) -> list[str]:
    with open(annotations_path, "r") as f:
        images = json.load(f)["images"]
    # Sorted so the split does not depend on set iteration order
    all_tows = sorted({get_tow(image["file_name"]) for image in images})
    train_tows, _ = train_test_split(
        all_tows,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_tows


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    aggregated_annotations_path = processing_dir / settings.coco_file_name
    aggregated_images_path = processing_dir / settings.images_folder_name
    aggregate_clips(
        raw_download_path, aggregated_annotations_path, aggregated_images_path
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
    train_tows = set(get_list_of_tows_to_include_in_train_set(mapped_annotations_path))
    print(f"Train tows: {sorted(train_tows)}")
    should_the_image_be_included_in_train_set = (
        lambda image_name: get_tow(image_name) in train_tows
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
