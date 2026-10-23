"""
Seatizen Atlas image dataset (instance segmentation subset)
Source: https://zenodo.org/records/13951614 (version 1.1.0 of https://doi.org/10.5281/zenodo.12819156)
    The latest version of the record (https://zenodo.org/records/20459523) only has image-level
    multilabel CSVs, so version 1.1.0 is used: it has the COCO instance segmentation file and
    images.zip.
Split logic: By site (second token of the file name, e.g. 20210830_MYT_scuba_...). Mayotte (MYT)
    -> val; Reunion (Hermitage), Europa Island and Mauritius -> train.
Category mapping:
  fish: Fish
  non-fish: Sea_cucumber, Sea_urchin, Turtle, Other_starfish
  discarded: Giant_clam, Sea_anemone, Sponges (sessile); Homo_Sapiens, Human_object, Waste,
      Trample, Useless; all coral, algae, seagrass and seafloor classes
Images: the 1,200 instance-segmentation images (out of 14,493 in images.zip). Boxes come from the
    COCO bbox of each instance polygon. Images without fish / non-fish are kept as background.
    One image whose file size differs from its annotated size is dropped (see build_coco).
    images.zip is 35 GB, so only these 1,200 images (~0.6 GB) are read from it with HTTP Range
    requests instead of downloading the whole zip.
"""

import io
import json
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    add_dataset_shortname_prefix_to_image_names,
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "seatizen_atlas"
RECORD_URL = "https://zenodo.org/api/records/13951614/files"
ANNOTATIONS_URL = f"{RECORD_URL}/20241018_segmentation_annotations.json/content"
IMAGES_ZIP_URL = f"{RECORD_URL}/images.zip/content"
ZIP_IMAGES_DIR = "2024_dataset"
# Every source category is listed; unlisted categories are discarded
CATEGORIES_FILTER = {
    "Fish": "fish",
    "Sea_cucumber": "non-fish",
    "Sea_urchin": "non-fish",
    "Turtle": "non-fish",
    "Other_starfish": "non-fish",
}
VAL_SITES = {"MYT"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_annotations_path = raw_download_path / "20241018_segmentation_annotations.json"
raw_images_path = raw_download_path / "images"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "seatizen_atlas_mapped_annotations.json"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP Range requests, with a read-ahead buffer."""

    def __init__(self, url: str, read_ahead: int = 2 * 1024 * 1024):
        self.url = url
        self.read_ahead = read_ahead
        self.session = requests.Session()
        self.size = int(self.session.head(url, allow_redirects=True).headers["Content-Length"])
        self.pos = 0
        self.buffer_start = 0
        self.buffer = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        # Zenodo allows 133 requests per minute. Pause when few are left, and wait a full
        # minute after a 429 (its Retry-After is often 0); 429s don't count as failed attempts.
        attempt = 0
        while attempt < 10:
            try:
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120
                )
                if response.status_code == 429:
                    print("Rate limited by Zenodo; waiting 60 s")
                    time.sleep(60)
                    continue
                if int(response.headers.get("X-RateLimit-Remaining", 100)) < 5:
                    reset_in = int(response.headers.get("X-RateLimit-Reset", 0)) - time.time()
                    time.sleep(min(max(reset_in, 5), 60) + 1)
                response.raise_for_status()
                assert response.status_code == 206, "Server ignored the Range header"
                assert len(response.content) == end - start
                return response.content
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                attempt += 1
                print(f"Connection error ({e}); retrying in 30 s")
                time.sleep(30)
        raise RuntimeError(f"Could not read bytes {start}-{end} of {self.url}")

    def read(self, n=-1):
        if n is None or n < 0 or n > self.size - self.pos:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = self.pos + n
        buffer_end = self.buffer_start + len(self.buffer)
        if not (self.buffer_start <= self.pos and end <= buffer_end):
            self.buffer_start = self.pos
            self.buffer = self._get(self.pos, min(self.size, self.pos + max(n, self.read_ahead)))
        offset = self.pos - self.buffer_start
        data = self.buffer[offset:offset + n]
        self.pos += len(data)
        return data


def download_data():
    raw_images_path.mkdir(parents=True, exist_ok=True)
    download_file_resumable(ANNOTATIONS_URL, raw_annotations_path)

    with open(raw_annotations_path) as f:
        file_names = [image["file_name"] for image in json.load(f)["images"]]
    missing = [name for name in file_names if not (raw_images_path / name).exists()]
    if not missing:
        print(f"All {len(file_names)} images already downloaded: {raw_images_path}")
        return

    # Read only the annotated images from the 35 GB zip, in zip order
    print(f"Reading {len(missing)} images from {IMAGES_ZIP_URL}")
    with zipfile.ZipFile(HttpRangeFile(IMAGES_ZIP_URL)) as zf:
        infos = sorted(
            (zf.getinfo(f"{ZIP_IMAGES_DIR}/{name}") for name in missing),
            key=lambda info: info.header_offset,
        )
        for info in tqdm(infos):
            target_path = raw_images_path / Path(info.filename).name
            partial = target_path.with_name(target_path.name + ".part")
            partial.write_bytes(zf.read(info))  # zipfile checks the CRC
            partial.rename(target_path)


def build_coco():
    """
    Write a COCO file with boxes only (no polygons), 1-indexed image and annotation ids, and the
    site of each image in "site".
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    with open(raw_annotations_path) as f:
        source = json.load(f)

    images = []
    dropped_image_ids = set()
    for image in tqdm(source["images"]):
        source_path = raw_images_path / image["file_name"]
        with Image.open(source_path) as im:
            assert im.getexif().get(274, 1) == 1, f"EXIF-rotated image: {image['file_name']}"
            if im.size != (image["width"], image["height"]):
                # 20210218_REU-Hermitage_scuba_4_00563.jpg is 1920x1080 in the zip but annotated
                # as 3840x2160, and its box (scaled or not) matches no animal: drop it
                print(f"Dropping {image['file_name']}: size {im.size} != annotated "
                      f"{(image['width'], image['height'])}")
                dropped_image_ids.add(image["id"])
                continue
        target_path = coco_images_path / image["file_name"]
        if not target_path.exists():
            target_path.write_bytes(source_path.read_bytes())
        images.append({
            "id": image["id"] + 1,
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "site": image["file_name"].split("_")[1],
        })

    annotations = []
    for annotation in source["annotations"]:
        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0 or annotation["image_id"] in dropped_image_ids:
            continue
        annotations.append({
            "id": annotation["id"] + 1,
            "image_id": annotation["image_id"] + 1,
            "category_id": annotation["category_id"],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    categories = [{"id": c["id"], "name": c["name"]} for c in source["categories"]]

    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)} (of {len(source['annotations'])})")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    build_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    # By site: Mayotte -> val, all other sites -> train
    with open(mapped_annotations_path) as f:
        train_images = {
            image["file_name"]
            for image in json.load(f)["images"]
            if image["site"] not in VAL_SITES
        }

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
