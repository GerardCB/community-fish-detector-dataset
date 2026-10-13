"""
Salmon Computer Vision Dataset
Source: https://github.com/Salmon-Computer-Vision/salmon-computer-vision
    (Dropbox folder linked from the README, file yolov6_salmon.tar.gz, ~107 GB)
Split logic: By video, reproducible (the video name is the frame filename without the
    "_frame_NNNNNN" suffix)
Category mapping:
  fish: all 16 classes (salmon-river fish; Lan prey, Lampray and Cutthroat have no boxes)
  non-fish: none
  discard: none

Every frame of every video is labelled, including frames without fish (~69% of frames),
and all of them are kept. Each COCO image entry carries extra fields so that consecutive
frames can be dropped later:
  video: video name (unique per clip)
  frame_index: frame number within the video
  video_start: recording start time parsed from the video name (ISO format; for the
      hour-long continuous recordings only the hour is known)
  camera: camera name parsed from the video name (e.g. right_bank_underwater), or None

The tarball is read in a single streaming pass: the YOLO labels come first in the archive,
then the images, which are written directly to the processing folder with the dataset
prefix, so the 107 GB archive is never fully extracted to disk. The archive is kept in
data/raw after processing; delete it by hand once the dataset is built.
"""

import io
import json
import re
import tarfile
import time
from pathlib import Path

import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "salmon_computer_vision"
SOURCE_URL = (
    "https://www.dropbox.com/scl/fo/rr8ht49arklvda1l6suuf/AKUVdaJv77b3gC9YpPvJSLE/"
    "yolov6_salmon.tar.gz?rlkey=u3yqhlpwkbxqtdhqwounb2cds&dl=1"
)
# YOLO class ids 0-15, in order (from combined_bear-kitwanga.yaml and the COCO files in the archive)
SOURCE_CLASSES = [
    "Coho",
    "Bull",
    "Rainbow",
    "Sockeye",
    "Pink",
    "Whitefish",
    "Chinook",
    "Shiner",
    "Pikeminnow",
    "Chum",
    "Steelhead",
    "Jack Chinook",
    "Juvenile Salmonoid",
    "Lan prey",
    "Lampray",
    "Cutthroat",
]
CATEGORIES_FILTER = {
    "Coho": "fish",
    "Bull": "fish",
    "Rainbow": "fish",
    "Sockeye": "fish",
    "Pink": "fish",
    "Whitefish": "fish",
    "Chinook": "fish",
    "Shiner": "fish",
    "Pikeminnow": "fish",
    "Chum": "fish",
    "Steelhead": "fish",
    "Jack Chinook": "fish",
    "Juvenile Salmonoid": "fish",
    "Lan prey": "fish",  # lamprey (misspelt), no boxes
    "Lampray": "fish",  # lamprey (misspelt), no boxes
    "Cutthroat": "fish",  # no boxes
}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
archive_path = raw_data_dir / "yolov6_salmon.tar.gz"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
raw_annotations_path = processing_dir / "salmon_computer_vision_coco.json"
mapped_annotations_path = processing_dir / "salmon_computer_vision_mapped_annotations.json"
images_path = processing_dir / settings.images_folder_name

VIDEO_NAME_PATTERNS = [
    # 07-15-2020_18-48-46_m_left_bank_underwater
    (
        re.compile(r"^(\d\d)-(\d\d)-(\d{4})_(\d\d)-(\d\d)-(\d\d)(?:_(.*))?$"),
        lambda m: (f"{m[3]}-{m[1]}-{m[2]}T{m[4]}:{m[5]}:{m[6]}", m[7]),
    ),
    # 10-07-2019_15_c_right_bank_underwater (hour-long continuous recording)
    (
        re.compile(r"^(\d\d)-(\d\d)-(\d{4})_(\d\d)_(.*)$"),
        lambda m: (f"{m[3]}-{m[1]}-{m[2]}T{m[4]}", m[5]),
    ),
    # 2019-08-10_11-55-45_right_bank
    (
        re.compile(r"^(\d{4})-(\d\d)-(\d\d)_(\d\d)-(\d\d)-(\d\d)(?:_(.*))?$"),
        lambda m: (f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}", m[7]),
    ),
]


def download_data(max_attempts: int = 20):
    """
    Download the tarball from Dropbox. Dropbox drops long connections, so a broken
    download is resumed from the partial file (also across reruns).
    """
    if archive_path.exists() or raw_annotations_path.exists():
        print(f"Data already downloaded: {archive_path}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".part")

    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading {SOURCE_URL} to {archive_path} (attempt {attempt}, resuming at {offset} bytes)")
        try:
            with requests.get(SOURCE_URL, headers=headers, stream=True, timeout=60) as response:
                if response.status_code == 416:  # nothing left to download
                    break
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(
                        "Server ignored the Range header; delete the .part file and rerun"
                    )
                total = offset + int(response.headers.get("content-length", 0))
                with (
                    open(partial, "ab") as f,
                    tqdm(total=total, initial=offset, unit="B", unit_scale=True, unit_divisor=1024) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Connection broken: {e}; retrying in 30 s")
            time.sleep(30)
            continue
        if partial.stat().st_size == total:
            break
    else:
        raise RuntimeError(f"Download incomplete after {max_attempts} attempts; rerun to resume")

    partial.rename(archive_path)


def parse_video_name(video: str) -> tuple[str | None, str | None]:
    """Return (video_start, camera) from a video name."""
    for pattern, parse in VIDEO_NAME_PATTERNS:
        match = pattern.match(video)
        if match:
            video_start, camera = parse(match)
            return video_start, re.sub(r"^[mc]_", "", camera) if camera else None
    return None, None


def yolo_to_coco_boxes(label_text: str, width: int, height: int) -> list[tuple[int, list[float]]]:
    boxes = []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, w, h = map(float, line.split())
        # Clip to the image: rounding in the labels can push edges ~0.001 px outside
        x0, y0 = max((cx - w / 2) * width, 0), max((cy - h / 2) * height, 0)
        x1, y1 = min((cx + w / 2) * width, width), min((cy + h / 2) * height, height)
        bbox = [round(v, 2) for v in (x0, y0, x1 - x0, y1 - y0)]
        boxes.append((int(class_id), bbox))
    return boxes


def extract_images_and_build_annotations():
    """
    Stream the tarball once: collect the YOLO labels (stored first in the archive), then
    write each image to the processing folder with the dataset prefix and add it to a
    COCO file with the source categories (1-indexed: YOLO class id + 1).
    """
    if raw_annotations_path.exists():
        print(f"Annotations already built: {raw_annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    existing_images = set(p.name for p in images_path.iterdir())

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(SOURCE_CLASSES)],
    }
    labels = {}  # frame stem -> YOLO label text

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in tqdm(archive, desc="Reading archive", unit=" files"):
            if not member.isfile():
                continue
            parts = Path(member.name).parts  # export_.../{labels,images}/{split}/{frame}
            if len(parts) != 4:
                continue  # labels/*.cache, annotations/*.json
            folder, stem, suffix = parts[1], Path(parts[3]).stem, Path(parts[3]).suffix

            if folder == "labels" and suffix == ".txt":
                labels[stem] = archive.extractfile(member).read().decode()
            elif folder == "images":
                data = archive.extractfile(member).read()
                width, height = Image.open(io.BytesIO(data)).size

                file_name = f"{DATASET_SHORTNAME}_{parts[3]}"
                if file_name not in existing_images:
                    partial = images_path / f"{file_name}.part"
                    partial.write_bytes(data)
                    partial.rename(images_path / file_name)

                video, frame_index = re.match(r"(.*)_frame_(\d+)$", stem).groups()
                video_start, camera = parse_video_name(video)
                image_id = len(coco["images"]) + 1
                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "video": video,
                        "frame_index": int(frame_index),
                        "video_start": video_start,
                        "camera": camera,
                    }
                )

                for class_id, bbox in yolo_to_coco_boxes(labels.pop(stem, ""), width, height):
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": class_id + 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )

    if labels:
        print(f"WARNING: {len(labels)} label files without an image, e.g. {next(iter(labels))}")
    print(f"Built {len(coco['images'])} images and {len(coco['annotations'])} annotations")

    partial = raw_annotations_path.with_name(raw_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco, f)
    partial.rename(raw_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    extract_images_and_build_annotations()

    # File names are already prefixed with DATASET_SHORTNAME during extraction
    map_annotations_to_fish_and_non_fish(
        raw_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def dataset_splitting():
    # Split by video, reproducible: sorted video names + fixed random_state
    with open(mapped_annotations_path) as f:
        image_to_video = {image["file_name"]: image["video"] for image in json.load(f)["images"]}
    train_videos, _ = train_test_split(
        sorted(set(image_to_video.values())),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_videos = set(train_videos)

    should_the_image_be_included_in_train_set = (
        lambda image_path: image_to_video[Path(image_path).name] in train_videos
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
