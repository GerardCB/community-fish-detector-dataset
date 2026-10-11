"""
DeepFish Dataset
Source: http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar
Split logic: By habitat (the first filename token). Habitats with fish and habitats with
    only empty frames are split separately (train_test_split on sorted ids)
Category mapping:
  fish: fish (every connected component of the segmentation masks)
  non-fish: none
  discard: mask_noise (components smaller than MIN_COMPONENT_AREA pixels: 1-20 px
    specks in the masks, not fish)

This dataset includes count, classification, and segmentation labels; we only use
the segmentation labels, reducing them to bounding boxes. Segmentation labels are
stored as mask images, so we parse connected components from the images. The
"empty" segmentation frames (no fish) are kept as background images.
"""

import json
import shutil
from pathlib import Path

import cv2
from skimage import measure
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    CompressionType,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "deepfish"
SOURCE_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"
CATEGORIES_FILTER = {
    "fish": "fish",
    # "mask_noise": discarded (1-20 px specks in the masks, not fish)
}

# Mask components smaller than this (in pixels) are annotation noise; the
# smallest real fish are ~5,000 px in these 1920x1080 frames.
MIN_COMPONENT_AREA = 100
SOURCE_CATEGORIES = [{"id": 1, "name": "fish"}, {"id": 2, "name": "mask_noise"}]
SEGMENTATION_SUBSETS = ["valid", "empty"]

settings = Settings()


def download_data(download_path: Path):
    """Download and extract the DeepFish dataset."""
    if (download_path / "DeepFish" / "Segmentation").exists():
        print(f"Data already exists: {download_path}")
        return
    download_and_extract(
        download_path, SOURCE_URL, DATASET_SHORTNAME, CompressionType.TAR
    )


def get_boxes_from_mask_image(mask_file: Path) -> list[tuple[list[int], int]]:
    """
    Load a binary mask, find connected components, and return (COCO bbox, area) per component.
    """
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    labels = measure.label(binary, connectivity=2)

    boxes = []
    for region in measure.regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        coco_bbox = [min_col, min_row, max_col - min_col, max_row - min_row]
        boxes.append((coco_bbox, int(region.area)))
    return boxes


def create_coco_dataset(
    download_path: Path, images_path: Path, coco_dataset_file: Path
) -> Path:
    """Convert the segmentation masks (valid + empty frames) to COCO and copy the images."""
    if coco_dataset_file.exists():
        print(f"COCO dataset already exists: {coco_dataset_file}")
        return coco_dataset_file

    segmentation_base = download_path / "DeepFish" / "Segmentation"

    images = []
    annotations = []
    for subset in SEGMENTATION_SUBSETS:
        image_dir = segmentation_base / "images" / subset
        mask_dir = segmentation_base / "masks" / subset
        image_files = sorted(image_dir.glob("*.jpg"))
        mask_files = sorted(mask_dir.glob("*.png"))
        print(f"{subset}: {len(image_files)} images, {len(mask_files)} masks")
        assert [p.stem for p in image_files] == [p.stem for p in mask_files], (
            f"Images and masks do not match in {subset}"
        )

        for image_file, mask_file in tqdm(
            zip(image_files, mask_files), total=len(image_files)
        ):
            im_cv = cv2.imread(str(image_file))
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": image_file.name,
                "height": im_cv.shape[0],
                "width": im_cv.shape[1],
            })
            shutil.copy2(image_file, images_path / image_file.name)

            for bbox, area in get_boxes_from_mask_image(mask_file):
                category_id = 1 if area >= MIN_COMPONENT_AREA else 2
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })

    print(f"Created {len(images)} images and {len(annotations)} annotations")

    coco_data = {
        "info": {},
        "categories": SOURCE_CATEGORIES,
        "annotations": annotations,
        "images": images,
    }
    with open(coco_dataset_file, "w") as f:
        json.dump(coco_data, f, indent=1)

    print(f"COCO dataset saved to {coco_dataset_file}")
    return coco_dataset_file


def get_habitat(image_filename: str) -> str:
    return remove_dataset_shortname_prefix_from_image_filename(
        Path(image_filename).name, DATASET_SHORTNAME
    ).split("_")[0]


def get_list_of_habitats_to_include_in_train_set(annotations_path: Path) -> list[str]:
    """
    Split habitats with fish and habitats with only empty frames separately, so the
    val set does not end up with mostly empty habitats.
    """
    with open(annotations_path) as f:
        coco_data = json.load(f)
    image_id_to_habitat = {
        image["id"]: get_habitat(image["file_name"]) for image in coco_data["images"]
    }
    habitats_with_fish = sorted(
        {image_id_to_habitat[a["image_id"]] for a in coco_data["annotations"]}
    )
    habitats_only_empty = sorted(set(image_id_to_habitat.values()) - set(habitats_with_fish))

    train_habitats = []
    for habitats in [habitats_with_fish, habitats_only_empty]:
        train, val = train_test_split(
            habitats,
            test_size=settings.train_val_split_ratio,
            random_state=settings.random_state,
        )
        print(f"Train habitats ({len(train)}): {sorted(train)}; val habitats: {sorted(val)}")
        train_habitats.extend(train)
    return train_habitats


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

    mapped_annotations_path = processing_dir / "deepfish_mapped_annotations.json"
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
    train_habitats = get_list_of_habitats_to_include_in_train_set(mapped_annotations_path)
    should_the_image_be_included_in_train_set = (
        lambda image_filename: get_habitat(image_filename) in train_habitats
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
