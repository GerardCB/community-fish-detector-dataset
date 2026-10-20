"""
OBSEA 2013-2014 Dataset (PANGAEA)
Source: https://doi.org/10.1594/PANGAEA.946149 (annotation table, plus the images zip that
    needs a PANGAEA login: https://download.pangaea.de/dataset/946149/allfiles.zip)
Split logic: By calendar month. train_test_split on the 24 sorted months (YYYYMM from the
    image filename) puts 5 whole months in val, so images taken 30 minutes apart never end up
    on both sides.
Category mapping:
  fish: all 29 fish taxa, plus "Unknown fish"
  non-fish: none
  discard: none

Still images from the OBSEA seafloor observatory (20 m, off Vilanova i la Geltru, Spain),
roughly one every 30 minutes, day and night: a Sony SNC-RZ25N camera (640x480) from 2013-01-10
to 2014-12-11 and an Axis P1346-E (2048x1536) until 2014-12-31. The annotation table has one
row per box with the 4 corners of a rectangle that can be rotated (36% of the boxes by 10
degrees or more). Each box becomes the axis-aligned box around the ellipse inscribed in that
rectangle, clipped to the image: fish are roughly elliptical, and the box around the corners
of a rectangle rotated by 30-45 degrees is 1.6-2x larger than the fish. For unrotated boxes
both are the same. Boxes listed twice (164, mostly once with a species and once as "Unknown
fish") are kept once, and boxes that are 1 pixel or less wide or high after clipping (102)
are dropped. Photos whose JPEG cannot be fully decoded are skipped with their boxes: 2 photos
of the zip are truncated (20131025_013028z and 20131121_070025z, 360 and 408 of their 480 rows
blank), leaving 14,950 images and 69,649 boxes.

Known issue: the annotations are incomplete. A manual count on 40 random images found 176
boxes and 251 clearly visible fish without a box (plus 98 possible ones): about 60% of the
visible fish are not annotated (95% bootstrap CI over images 40-71%). 28 of the 40 images have
at least one unboxed fish. Most of them are in schools, where only a few fish are boxed (5
images hold 76% of the missing fish); elsewhere about 1 in 3 visible fish has no box. Until
the dataset is cleaned, these images teach a detector that many fish are background.

Not the same images as the `obsea` dataset (Zenodo 14888328), which is from 2023-2024 and
other cameras.
"""

import csv
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    map_annotations_to_fish_and_non_fish,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "obsea_2013_2014"
TABLE_URL = "https://doi.pangaea.de/10.1594/PANGAEA.946149?format=textfile"
TABLE_NAME = "obsea_fish_2013_14.tab"
IMAGES_ZIP_URL = "https://download.pangaea.de/dataset/946149/allfiles.zip"
IMAGES_ZIP_NAME = "obsea_fish_2013_14_allfiles.zip"
EXPECTED_ROWS = 69917
EXPECTED_IMAGES = 14952

FISH_CLASSES = [
    "Apogon sp.",
    "Atherina sp.",
    "Boops boops",
    "Chromis chromis",
    "Conger conger",
    "Coris julis",
    "Dentex dentex",
    "Diplodus annularis",
    "Diplodus cervinus",
    "Diplodus puntazzo",
    "Diplodus sargus",
    "Diplodus vulgaris",
    "Epinephelus marginatus",
    "Oblada melanura",
    "Pagellus sp.",
    "Pagrus pagrus",
    "Sarpa salpa",
    "Sciaena umbra",
    "Scorpaena sp.",
    "Seriola dumerili",
    "Serranus cabrilla",
    "Sparus aurata",
    "Spicara maena",
    "Spondyliosoma cantharus",
    "Symphodus cinereus",
    "Symphodus mediterraneus",
    "Symphodus tinca",
    "Thalassoma pavo",
    "Trachurus sp.",
    # 60 random crops checked: mostly small, distant or night-time fish, none clearly anything else
    "Unknown fish",
]
# Every source category must be listed here explicitly; unlisted categories are discarded.
CATEGORIES_FILTER = {name: "fish" for name in FISH_CLASSES}

MIN_BOX_SIZE = 1  # pixels; boxes this narrow or low after clipping are dropped

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
table_path = raw_data_dir / TABLE_NAME
source_images_dir = raw_data_dir / "images"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "obsea_2013_2014_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def download_data():
    """
    Download the annotation table, and extract the images zip, which has to be downloaded by
    hand because PANGAEA needs a login for it.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    if not table_path.exists():
        download_file(TABLE_URL, table_path)

    extracted_marker = raw_data_dir / ".images_extracted"
    if extracted_marker.exists():
        print(f"Images already extracted at {source_images_dir}")
        return

    zip_path = raw_data_dir / IMAGES_ZIP_NAME
    if not zip_path.exists():
        raise SystemExit(
            f"The images need a (free) PANGAEA account. Sign in at "
            f"https://www.pangaea.de/user/login.php, download {IMAGES_ZIP_URL} in the same "
            f"browser and save it as {zip_path}, then rerun."
        )

    print(f"Extracting {zip_path}...")
    source_images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zipf:
        for member in tqdm(zipf.infolist(), desc="Extracting images"):
            if member.is_dir() or not member.filename.lower().endswith(".jpg"):
                continue
            target = source_images_dir / Path(member.filename).name
            with zipf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    extracted_marker.touch()


def read_annotation_table() -> list[dict]:
    """Rows of the PANGAEA table (after its "/* ... */" header) as dicts."""
    text = table_path.read_text(encoding="utf-8")
    lines = text.split("*/\n", 1)[1].splitlines()
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{table_path} has {len(rows)} rows, expected {EXPECTED_ROWS}")
    return rows


def polygon_to_box(row: dict, image_width: int, image_height: int) -> list[float] | None:
    """
    Axis-aligned [x, y, w, h] around the ellipse inscribed in the (possibly rotated)
    rectangle given by the 4 consecutive corners of the row, clipped to the image.
    """
    xs = [float(row[f"bboxx{i} [pixel]"]) for i in range(1, 5)]
    ys = [float(row[f"bboxy{i} [pixel]"]) for i in range(1, 5)]
    center_x, center_y = sum(xs) / 4, sum(ys) / 4
    # Half side vectors of the rectangle; the inscribed ellipse reaches this far along x and y
    half_side_1 = ((xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2)
    half_side_2 = ((xs[2] - xs[1]) / 2, (ys[2] - ys[1]) / 2)
    half_width = math.hypot(half_side_1[0], half_side_2[0])
    half_height = math.hypot(half_side_1[1], half_side_2[1])

    x_min, y_min = max(0.0, center_x - half_width), max(0.0, center_y - half_height)
    x_max = min(float(image_width), center_x + half_width)
    y_max = min(float(image_height), center_y + half_height)
    if x_max - x_min <= MIN_BOX_SIZE or y_max - y_min <= MIN_BOX_SIZE:
        return None
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_obsea_2013_2014_to_coco():
    """
    Convert the annotation table to COCO with the source species as categories, copying the
    images with the dataset prefix.
    """
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    rows = read_annotation_table()
    rows_by_image = {}
    for row in rows:
        rows_by_image.setdefault(row["IMAGE"], []).append(row)
    if len(rows_by_image) != EXPECTED_IMAGES:
        raise ValueError(f"{len(rows_by_image)} images in the table, expected {EXPECTED_IMAGES}")

    missing = [name for name in rows_by_image if not (source_images_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images of the table are not in {source_images_dir}, e.g. {missing[:3]}"
        )

    species = sorted({row["Species"] for row in rows})
    category_ids = {name: category_id for category_id, name in enumerate(species, start=1)}
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_ids[name], "name": name} for name in species],
    }
    dropped = Counter()
    images_path.mkdir(parents=True, exist_ok=True)

    for image_name, image_rows in tqdm(
        sorted(rows_by_image.items()), total=len(rows_by_image), desc="Converting to COCO"
    ):
        source_image_path = source_images_dir / image_name
        with Image.open(source_image_path) as image:
            image_width, image_height = image.size
            # 2 JPEGs of the zip are truncated (most rows blank); they are skipped with their boxes
            try:
                image.load()
            except OSError as error:
                print(f"Skipping {image_name}: {error}")
                dropped["images that cannot be decoded"] += 1
                dropped["boxes on images that cannot be decoded"] += len(image_rows)
                continue

        image_id = len(coco_data["images"]) + 1
        file_name = f"{DATASET_SHORTNAME}_{image_name}"
        shutil.copy2(source_image_path, images_path / file_name)
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_width,
                "height": image_height,
                "datetime": image_rows[0]["Date/Time"],
                "camera": image_name.split("_camera_")[1].rsplit(".", 1)[0],
            }
        )

        # 164 boxes are listed twice, 163 of them with two labels (mostly a species and
        # "Unknown fish"); all labels are fish, so only the first row is kept
        seen = set()
        for row in image_rows:
            corners = tuple(row[f"bbox{axis}{i} [pixel]"] for i in range(1, 5) for axis in "xy")
            if corners in seen:
                dropped["same box listed twice"] += 1
                continue
            seen.add(corners)

            bbox = polygon_to_box(row, image_width, image_height)
            if bbox is None:
                dropped["too small after clipping"] += 1
                continue
            coco_data["annotations"].append(
                {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_ids[row["Species"]],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )

    print(
        f"Images: {len(coco_data['images'])}, boxes: {len(coco_data['annotations'])}, "
        f"dropped: {dict(dropped)}"
    )
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco_data, f)
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_obsea_2013_2014_to_coco()
    map_annotations_to_fish_and_non_fish(
        source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def get_month(image_filename: str) -> str:
    """Month of a photo: "obsea_2013_2014_20130110_105250z_OBSEA_camera_....jpg" -> "201301"."""
    return remove_dataset_shortname_prefix_from_image_filename(
        Path(image_filename).name, DATASET_SHORTNAME
    )[:6]


def get_validation_months() -> set[str]:
    with open(mapped_annotations_path) as f:
        months = sorted({get_month(image["file_name"]) for image in json.load(f)["images"]})
    _, validation_months = train_test_split(
        months, test_size=settings.train_val_split_ratio, random_state=settings.random_state
    )
    print(f"Validation months: {sorted(validation_months)}")
    return set(validation_months)


def dataset_splitting():
    validation_months = get_validation_months()
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_month(image_path) not in validation_months
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
