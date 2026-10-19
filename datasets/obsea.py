"""
OBSEA Dataset
Source: https://doi.org/10.5281/zenodo.14888328 (obsea_dataset_v4.1.zip, YOLO boxes)
Split logic: By camera. Cameras AIPC608UW_10_167 and C4k0193 go to val; all other
    cameras and the videos/images without a camera id go to train.
Category mapping:
  fish: all fish species, plus "unknown" and "unknown sp" (crops checked: nearly all fish)
  non-fish: Cranc (crab), Octopus vulgaris, Rhizostoma pulmo, Chrysaora hysoscella,
    Asteroidea
  discard: Diver, Delphinus (dead) (a carcass on the seabed), Posidonia oceanica (shot)
    (seagrass), Gateropodae (small snail shells)

The YOLO class ids follow the order of the class names in obsea_dataset.json, which also
holds the box count per class; the conversion checks those counts. Images with no boxes
left are kept.
"""

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "obsea"
DATA_URL = "https://zenodo.org/records/14888328/files/obsea_dataset_v4.1.zip?download=1"
SOURCE_ZIP_NAME = "obsea_dataset_v4.1.zip"
SOURCE_ZIP_MD5 = "7a2012d7a39fb42d155b29fc67d8753f"
SOURCE_FOLDER_NAME = "obsea_dataset_v4.1"

FISH_CLASSES = [
    "Aetomylaeus bovinus",
    "Apogon imberbis",
    "Apogon imberbis (night)",
    "Apogon sp.",
    "Balistes capriscus",
    "Chromis chromis",
    "Chromis chromis (alev.)",
    "Chromis chromis (back)",
    "Coris julis",
    "Coris julis (back)",
    "Coris julis (m)",
    "Dactylopterus volitans",
    "Dentex dentex",
    "Dentex dentex (back)",
    "Dentex dentex (night)",
    "Diplodus annularis",
    "Diplodus cervinus",
    "Diplodus cervinus (alev.)",
    "Diplodus cervinus (back)",
    "Diplodus cervinus (night)",
    "Diplodus puntazzo",
    "Diplodus puntazzo (night)",
    "Diplodus sargus",
    "Diplodus sargus (back)",
    "Diplodus sargus (night)",
    "Diplodus vulgaris",
    "Diplodus vulgaris (night)",
    "Epinephelus costae",
    "Epinephelus marginatus",
    "Mola mola",
    "Mullus surmuletus",
    "Mullus surmuletus (night)",
    "Muraena helena",
    "Muraena helena (alev.)",
    "Muraena helena (night)",
    "Myliobatidae",
    "Myliobatidae (back)",
    "Oblada melanura",
    "Oblada melanura (back)",
    "Oblada melanura (night)",
    "Parablennius gattorugine",
    "Sarpa salpa",
    "Sarpa salpa (night)",
    "Sciaena umbra",
    "Sciaena umbra (night)",
    "Scorpaena sp",
    "Seriola dumerili",
    "Seriola dumerili (back)",
    "Seriola dumerili (night)",
    "Serranus cabrilla",
    "Serranus scriba",
    "Serranus scriba (night)",
    "Sparidae",
    "Sparus aurata",
    "Sparus aurata (back)",
    "Spondyliosoma cantharus",
    "Symphodus mediterraneus",
    "Symphodus mediterraneus (f)",
    "Symphodus mediterraneus (m)",
    "Symphodus melanocercus",
    "Symphodus tinca (m)",
    "Thalassoma pavo",
    "Thalassoma pavo (f)",
    # Crops checked: nearly all fish (a few unclear boxes in "unknown sp")
    "unknown",
    "unknown sp",
]
NON_FISH_CLASSES = [
    "Asteroidea",  # starfish
    "Chrysaora hysoscella",  # compass jellyfish
    "Cranc",  # crab (Catalan)
    "Octopus vulgaris",
    "Rhizostoma pulmo",  # barrel jellyfish
]
# Discarded: Diver, Delphinus (dead), Posidonia oceanica (shot), Gateropodae
CATEGORIES_FILTER = {
    **{name: "fish" for name in FISH_CLASSES},
    **{name: "non-fish" for name in NON_FISH_CLASSES},
}

# Cameras assigned to the validation set
VALIDATION_GROUPS = {"AIPC608UW_10_167", "C4k0193"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
source_root = raw_data_dir / SOURCE_FOLDER_NAME
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "obsea_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download (resuming broken downloads), check and extract the zip from Zenodo."""
    extracted_marker = raw_data_dir / f".{SOURCE_FOLDER_NAME}_extracted"
    if extracted_marker.exists():
        print(f"Dataset already extracted at {source_root}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_dir / SOURCE_ZIP_NAME
    download_file_resumable(DATA_URL, zip_path)

    print(f"Checking MD5 of {zip_path}...")
    actual_md5 = md5sum(zip_path)
    if actual_md5 != SOURCE_ZIP_MD5:
        raise ValueError(
            f"MD5 mismatch for {zip_path}: expected {SOURCE_ZIP_MD5}, got {actual_md5}. "
            "Delete it and rerun."
        )

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zipf:
        zipf.extractall(raw_data_dir)
    extracted_marker.touch()


def load_source_class_names() -> tuple[list[str], dict[str, int]]:
    """
    Class names from obsea_dataset.json, in YOLO class id order, and the number of
    boxes per class that the file lists.
    """
    with open(source_root / "obsea_dataset.json") as f:
        box_counts = json.load(f)
    return list(box_counts), box_counts


def list_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    """Return {stem: path} for the files in folder with one of the given extensions."""
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }


def parse_yolo_label_file(
    label_path: Path,
    class_count: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Read a list of [class_id, x_center, y_center, box_width, box_height] bounding
    boxes from the .txt file [label_path].
    """
    rows = []
    with open(label_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"Malformed YOLO row at {label_path}:{line_number}: {line}"
                )

            try:
                class_id = int(fields[0])
                x_center, y_center, box_width, box_height = [
                    float(value) for value in fields[1:]
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Malformed YOLO values at {label_path}:{line_number}: {line}"
                ) from exc

            if not 0 <= class_id < class_count:
                raise ValueError(
                    f"Class id {class_id} outside [0, {class_count - 1}] at "
                    f"{label_path}:{line_number}"
                )

            if not (
                (0 <= x_center <= 1)
                and (0 <= y_center <= 1)
                and (0 < box_width <= 1)
                and (0 < box_height <= 1)
            ):
                raise ValueError(
                    f"YOLO coordinates outside normalized bounds at "
                    f"{label_path}:{line_number}: {line}"
                )

            rows.append((class_id, x_center, y_center, box_width, box_height))

    return rows


def yolo_box_to_coco_box(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    Convert a YOLO box (normalized x_center, y_center, w, h) to COCO (x_min, y_min, w, h)
    in pixels, clipped to the image.
    """
    x_min = (x_center - box_width / 2) * image_width
    y_min = (y_center - box_height / 2) * image_height
    x_max = (x_center + box_width / 2) * image_width
    y_max = (y_center + box_height / 2) * image_height

    x_min = max(0.0, min(float(image_width), x_min))
    y_min = max(0.0, min(float(image_height), y_min))
    x_max = max(0.0, min(float(image_width), x_max))
    y_max = max(0.0, min(float(image_height), y_max))

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def get_source_group(image_stem_or_filename: str) -> str:
    """
    Extract an OBSEA camera id from a filename, returns
    "unknown" if it can't find a known camera id.
    """
    stem = Path(image_stem_or_filename).stem
    if stem.startswith(f"{DATASET_SHORTNAME}_"):
        stem = stem[len(f"{DATASET_SHORTNAME}_") :]

    if "AIPC608UW_10_167" in stem:
        return "AIPC608UW_10_167"
    if "IPC608_8BC7_166" in stem:
        return "IPC608_8BC7_166"
    if "IPC608_8B64_166" in stem:
        return "IPC608_8B64_166"
    if "IPC608_8B64_165" in stem:
        return "IPC608_8B64_165"
    if "C4k0193" in stem:
        return "C4k0193"
    if "Mero" in stem or "Morena" in stem:
        return "Mero/Morena video"
    if "Video" in stem:
        return "Video/other"
    return "unknown"


def convert_obsea_to_coco():
    """
    Convert the YOLO labels to COCO with the source classes as categories (id = YOLO
    class id + 1), copying the images with the dataset prefix.
    """
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    class_names, expected_box_counts = load_source_class_names()
    source_images = list_files(source_root / "images", IMAGE_EXTENSIONS)
    source_labels = list_files(source_root / "labels", {".txt"})
    print(
        f"Read {len(class_names)} class names, {len(source_images)} images, "
        f"{len(source_labels)} label files"
    )

    missing_labels = sorted(set(source_images) - set(source_labels))
    missing_images = sorted(set(source_labels) - set(source_images))
    if missing_labels or missing_images:
        raise ValueError(
            "OBSEA image/label mismatch. "
            f"Images without labels: {len(missing_labels)}. "
            f"Labels without images: {len(missing_images)}."
        )

    images_path.mkdir(parents=True, exist_ok=True)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id + 1, "name": name} for class_id, name in enumerate(class_names)
        ],
    }
    box_counts = Counter()

    for image_id, (image_stem, source_image_path) in tqdm(
        enumerate(source_images.items(), start=1),
        total=len(source_images),
        desc="Converting YOLO to COCO",
    ):
        with Image.open(source_image_path) as image:
            image_width, image_height = image.size

        file_name = f"{DATASET_SHORTNAME}_{source_image_path.name}"
        shutil.copy2(source_image_path, images_path / file_name)
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_width,
                "height": image_height,
            }
        )

        rows = parse_yolo_label_file(source_labels[image_stem], len(class_names))
        for class_id, x_center, y_center, box_width, box_height in rows:
            bbox = yolo_box_to_coco_box(
                x_center, y_center, box_width, box_height, image_width, image_height
            )
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"Box collapsed to zero area in {source_image_path.name}")

            box_counts[class_names[class_id]] += 1
            coco_data["annotations"].append(
                {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )

    # The box counts in obsea_dataset.json confirm that the class names are in YOLO id order
    mismatched = {
        name: (count, box_counts[name])
        for name, count in expected_box_counts.items()
        if box_counts[name] != count
    }
    if mismatched:
        raise ValueError(f"Box counts differ from obsea_dataset.json: {mismatched}")

    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco_data, f)
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_obsea_to_coco()
    map_annotations_to_fish_and_non_fish(
        source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def dataset_splitting():
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_source_group(image_path) not in VALIDATION_GROUPS
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
