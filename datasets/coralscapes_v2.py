"""
CoralscapesV2 Dataset
Source: https://josauder.github.io/coralscapesv2/ (Hugging Face: josauder/coralscapesV2,
    "instances" configuration)
Split logic: By site. The sites the authors held out (their validation and test sites) go to
    val; site30, which is in both their train and validation splits, goes to train
Category mapping:
  fish: fish (one box per fish instance mask)
  non-fish: acanthaster planci (crown-of-thorns starfish), sea cucumber, other animal
      (mostly sea turtles, some starfish); boxes from connected components of the semantic
      mask, ignoring components under MIN_COMPONENT_PIXELS
  discard: sea urchin (no instance masks and urchins cluster, so one component often covers
      several urchins), giant clam, dead giant clam, anemone, feather worm, cirrhipathes,
      sponge, human, and all coral, substrate, equipment and background classes

Only the "instances" Parquet shards are downloaded (~6.9 GB): each row holds the image, the
95-class semantic mask, the fish instance ids and the site. Coralscapes v1 is not processed
separately: CoralscapesV2 contains all its images but one.
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage
from skimage import measure
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "coralscapes_v2"
HF_BASE_URL = "https://huggingface.co/datasets/josauder/coralscapesV2/resolve/main/"
SHARDS = (
    [f"instances/train-{i:05d}-of-00009.parquet" for i in range(9)]
    + ["instances/validation-00000-of-00001.parquet"]
    + [f"instances/test-{i:05d}-of-00003.parquet" for i in range(3)]
)
CATEGORIES_FILTER = {
    "fish": "fish",
    "acanthaster planci": "non-fish",
    "sea cucumber": "non-fish",
    "other animal": "non-fish",
    # "sea urchin" is discarded (clustered urchins merge into one component)
}

# Semantic class ids (id2label.json); fish instances are encoded as 25 * 2000 + index
FISH_CLASS_ID = 25
INSTANCE_LABEL_DIVISOR = 2000
# Animal classes without instance masks: boxes from connected components of the semantic mask
COMPONENT_CLASSES = {
    1: "acanthaster planci",
    55: "other animal",
    70: "sea cucumber",
    71: "sea urchin",
}
MIN_COMPONENT_PIXELS = 200  # smaller components are mostly slivers of occluded animals
# Sites in the authors' validation and test splits (site30 is in both train and validation)
VAL_SITES = {"site1", "site6", "site22", "site23", "site26", "site29", "site34"}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
raw_annotations_path = processing_dir / "coralscapes_v2_coco.json"
mapped_annotations_path = processing_dir / "coralscapes_v2_mapped_annotations.json"


def download_data():
    """Download the "instances" Parquet shards from Hugging Face (resumes broken downloads)."""
    for shard in SHARDS:
        shard_path = raw_data_dir / shard
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        download_file_resumable(HF_BASE_URL + shard, shard_path)


def fish_boxes(instance_ids: np.ndarray) -> list[tuple[list[int], int]]:
    """One (bbox, area) per fish instance."""
    fish = instance_ids // INSTANCE_LABEL_DIVISOR == FISH_CLASS_ID
    labels = np.where(fish, instance_ids % INSTANCE_LABEL_DIVISOR + 1, 0).astype(np.int32)
    areas = np.bincount(labels.ravel())
    boxes = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        boxes.append(([xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start], int(areas[index])))
    return boxes


def component_boxes(semantic: np.ndarray, class_id: int) -> list[tuple[list[int], int]]:
    """One (bbox, area) per connected component of a semantic class."""
    boxes = []
    for region in measure.regionprops(measure.label(semantic == class_id, connectivity=2)):
        if region.area < MIN_COMPONENT_PIXELS:
            continue
        y0, x0, y1, x1 = region.bbox
        boxes.append(([x0, y0, x1 - x0, y1 - y0], int(region.area)))
    return boxes


def build_coco_dataset():
    """Write each image to the processing folder and build a COCO file with source categories."""
    if raw_annotations_path.exists():
        print(f"COCO dataset already exists: {raw_annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    category_names = {FISH_CLASS_ID: "fish", **COMPONENT_CLASSES}
    category_ids = {class_id: i + 1 for i, class_id in enumerate(category_names)}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_ids[c], "name": n} for c, n in category_names.items()],
    }

    for shard in SHARDS:
        parquet_file = pq.ParquetFile(raw_data_dir / shard)
        for row_group in tqdm(range(parquet_file.num_row_groups), desc=shard):
            rows = parquet_file.read_row_group(
                row_group, columns=["image", "label", "instance_ids", "stem", "site"]
            ).to_pylist()
            for row in rows:
                image_bytes = row["image"]["bytes"]
                image = Image.open(io.BytesIO(image_bytes))
                file_name = f"{DATASET_SHORTNAME}_{row['stem']}.{image.format.lower()}"
                target = images_path / file_name
                if not target.exists():
                    target.write_bytes(image_bytes)

                image_id = len(coco["images"]) + 1
                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": image.width,
                        "height": image.height,
                        "site": row["site"],
                    }
                )

                semantic = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])))
                instance_ids = np.asarray(Image.open(io.BytesIO(row["instance_ids"]["bytes"])))
                boxes = [(FISH_CLASS_ID, box) for box in fish_boxes(instance_ids)]
                for class_id in COMPONENT_CLASSES:
                    boxes += [(class_id, box) for box in component_boxes(semantic, class_id)]

                for class_id, (bbox, area) in boxes:
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": category_ids[class_id],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )

    print(f"Built {len(coco['images'])} images and {len(coco['annotations'])} annotations")
    partial = raw_annotations_path.with_name(raw_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco, f)
    partial.rename(raw_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    # File names are prefixed with DATASET_SHORTNAME when the images are written
    build_coco_dataset()

    map_annotations_to_fish_and_non_fish(
        raw_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        image_to_site = {image["file_name"]: image["site"] for image in json.load(f)["images"]}

    should_the_image_be_included_in_train_set = (
        lambda image_path: image_to_site[Path(image_path).name] not in VAL_SITES
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
