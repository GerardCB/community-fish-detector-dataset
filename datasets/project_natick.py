"""
Project Natick Dataset
Source: https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip
Split logic: By chunks of FRAMES_PER_CHUNK consecutive frames of each video
    (train_test_split on video+chunk identifiers). The dataset has only 3 videos,
    one of them with 96% of the frames, so splitting by video is not possible, and
    splitting by single frame would put near-identical consecutive frames in both sets.
Category mapping:
  fish: Fish
  non-fish: none
  discard: Squid. Despite the label, these are arrow worms (chaetognaths): thin,
    translucent planktonic worms, mostly under ~3 cm. The subset is named
    "fish_arrow_worms_annotation".

Pascal VOC format annotations converted to COCO format. The release has two subsets,
"fish_arrow_worms_annotation" (1072 frames from one video) and "fish_annotations"
(46 frames from two other videos); both are used.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    extract_downloaded_file,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "project_natick"
DATA_URL = "https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip"
CATEGORIES_FILTER = {
    "Fish": "fish",
}
SUBSETS = ["fish_arrow_worms_annotation", "fish_annotations"]
FRAMES_PER_CHUNK = 25

settings = Settings()


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_path = data_dir / "data_release.zip"

    if not (data_dir / "data_release").exists():
        print("Downloading data...")
        download_file(DATA_URL, data_path)
        print("Extracting data...")
        extract_downloaded_file(data_path, data_dir)
    else:
        print("Data already downloaded and extracted")


def pascal_voc_to_coco(dataset_dir: Path, images_output_dir: Path, output_json: Path):
    """
    Converts the Pascal VOC subsets to a single COCO file and copies their images.
    The XML "filename" elements have no extension, so images are matched by the
    XML file stem instead.
    """
    if output_json.exists():
        print(f"COCO annotations already exist: {output_json}")
        return

    images_output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = {}

    for subset in SUBSETS:
        subset_dir = dataset_dir / subset
        for xml_path in sorted((subset_dir / "Annotations").glob("*.xml")):
            image_path = subset_dir / "JPEGImages" / f"{xml_path.stem}.jpg"
            root = ET.parse(xml_path).getroot()

            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": int(root.findtext("size/width")),
                    "height": int(root.findtext("size/height")),
                }
            )
            shutil.copy2(image_path, images_output_dir / image_path.name)

            for obj in root.findall("object"):
                label = obj.findtext("name")
                if label not in categories:
                    categories[label] = len(categories) + 1
                xmin, ymin, xmax, ymax = (
                    float(obj.findtext(f"bndbox/{key}"))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                )
                width, height = xmax - xmin, ymax - ymin
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": categories[label],
                        "bbox": [xmin, ymin, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in categories.items()],
    }
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Converted {len(images)} images and {len(annotations)} annotations")


def get_video_chunk(image_filename: str) -> str:
    """
    "20180602_23-41-08-0730-1430_frame_176.jpg" -> "20180602_23-41-08-0730-1430_7"
    """
    image_filename = remove_dataset_shortname_prefix_from_image_filename(
        Path(image_filename).name, DATASET_SHORTNAME
    )
    video, frame = Path(image_filename).stem.rsplit("_frame_", 1)
    return f"{video}_{int(frame) // FRAMES_PER_CHUNK}"


def get_list_of_chunks_to_include_in_train_set(annotations_path: Path) -> list[str]:
    with open(annotations_path, "r") as f:
        images = json.load(f)["images"]
    # Sorted so the split does not depend on set iteration order
    chunks = sorted({get_video_chunk(image["file_name"]) for image in images})
    train_chunks, _ = train_test_split(
        chunks,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_chunks


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_annotations_path = processing_dir / settings.coco_file_name
    pascal_voc_to_coco(
        raw_download_path / "data_release", coco_images_path, coco_annotations_path
    )

    mapped_annotations_path = processing_dir / "project_natick_mapped_annotations.json"
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
    train_chunks = set(get_list_of_chunks_to_include_in_train_set(mapped_annotations_path))
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_video_chunk(image_path) in train_chunks
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
