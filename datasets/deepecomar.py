"""
Labelled Mediterranean Fish Dataset (DEEPECOMAR)
Source: https://zenodo.org/records/7534425 (DEEPECOMAR_DATASET_YOLO.zip, Roboflow YOLOv5 export)
Images come from three sources, identified by filename:
    vivo_<YYYYMMDD>_<HHMMSS>M          fixed time-lapse camera on an underwater structure (2018-19)
    vlcsnap-<YYYY-MM-DD>-<time>        snapshots from videos recorded on 4 dates in 2021
    E01L_-<n>-_frame_<n>               frames of one stereo-camera video (left camera)
Roboflow appends "_jpg.rf.<hash>" to every name. 74 frames appear twice in train as identical
copies (same pixels and labels), so we keep one image per original name.
Split logic: By location. The authors' train/valid split is random per image, so it is
    ignored. Val = video snapshots from 2021-04-29 and 2021-05-03 (separate dives, ~20% of the
    images); train = everything else, including the whole fixed camera.
Category mapping:
  fish: all 22 named taxa, "Afish poly" (unidentified fish), "small" (fish < ~100 px^2)
  discard: "taca" (lens stain)
"""

import json
import re
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    CompressionType,
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "deepecomar"
DATA_URL = "https://zenodo.org/records/7534425/files/DEEPECOMAR_DATASET_YOLO.zip?download=1"
# YOLO class ids 0-24 in data.yaml order
CLASS_NAMES = [
    "Afish poly", "Chromis chromis", "Coris julis", "Dentex dentex", "Diplodus annularis",
    "Diplodus puntazzo", "Diplodus sargus", "Diplodus sp-", "Diplodus vulgaris",
    "Epinephelus marginatus", "Lithognathus mormyrus", "Mugilidae prob Chelon", "Mullus sp-",
    "Oblada melanura", "Pomatous salator", "Sarpa salpa", "Seriola dumerilii",
    "Serranus cabrilla", "Serranus scriba", "Sparus aurata", "Sphyraena sp-", "Symphodus sp-",
    "labrid unid-", "small", "taca",
]
CATEGORIES_FILTER = {
    "Afish poly": "fish",  # unidentified fish
    "Chromis chromis": "fish",
    "Coris julis": "fish",
    "Dentex dentex": "fish",
    "Diplodus annularis": "fish",
    "Diplodus puntazzo": "fish",
    "Diplodus sargus": "fish",
    "Diplodus sp-": "fish",
    "Diplodus vulgaris": "fish",
    "Epinephelus marginatus": "fish",
    "Lithognathus mormyrus": "fish",
    "Mugilidae prob Chelon": "fish",
    "Mullus sp-": "fish",
    "Oblada melanura": "fish",
    "Pomatous salator": "fish",  # Pomatomus saltatrix (bluefish)
    "Sarpa salpa": "fish",
    "Seriola dumerilii": "fish",
    "Serranus cabrilla": "fish",
    "Serranus scriba": "fish",
    "Sparus aurata": "fish",
    "Sphyraena sp-": "fish",
    "Symphodus sp-": "fish",
    "labrid unid-": "fish",  # unidentified Labridae
    "small": "fish",  # very small fish (< ~100 px^2), boxes of 1-20 px
    # "taca": discarded (lens stain)
}
VAL_SOURCES = ("vlcsnap-2021-04-29", "vlcsnap-2021-05-03")

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "DEEPECOMAR_DATASET_YOLO"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "deepecomar_mapped_annotations.json"


def download_data():
    if raw_data_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "DEEPECOMAR_DATASET_YOLO.zip"
    download_file_resumable(DATA_URL, zip_path)
    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def original_name(roboflow_path: Path) -> str:
    """vivo_20180820_093629M_jpg.rf.<hash>.jpg -> vivo_20180820_093629M.jpg"""
    stem = roboflow_path.name.split(".rf.")[0]
    return re.sub(r"_(jpe?g|JPE?G|png)$", "", stem) + ".jpg"


def yolo_to_coco():
    """Convert the YOLO labels of both source folders (train, valid) into one COCO dataset."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    # One image per original frame: Roboflow duplicates are identical copies
    image_paths = {}
    for image_path in sorted(raw_data_dir.glob("*/images/*.jpg")):
        image_paths.setdefault(original_name(image_path), image_path)
    n_files = len(list(raw_data_dir.glob("*/images/*.jpg")))
    print(f"{n_files} image files, {len(image_paths)} unique frames")

    images = []
    annotations = []
    for name, image_path in tqdm(sorted(image_paths.items())):
        with Image.open(image_path) as im:
            width, height = im.size
        shutil.copy2(image_path, coco_images_path / name)
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": name,
            "width": width,
            "height": height,
        })

        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            # YOLO format: class_id cx cy w h (normalized)
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bbox_w, bbox_h = w * width, h * height
            bbox_x, bbox_y = cx * width - bbox_w / 2, cy * height - bbox_h / 2
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(bbox_w * bbox_h, 2),
                "iscrowd": 0,
            })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)],
    }
    with open(coco_annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    yolo_to_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    def should_the_image_be_included_in_train_set(image_path: str) -> bool:
        name = Path(image_path).name.removeprefix(f"{DATASET_SHORTNAME}_")
        return not name.startswith(VAL_SOURCES)

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
