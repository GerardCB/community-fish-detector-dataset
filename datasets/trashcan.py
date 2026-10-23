"""
TrashCan 1.0 (deep-sea ROV video frames from the JAMSTEC J-EDI archive, annotated for marine debris, ROVs and fauna)
Source: https://doi.org/10.13020/g1gx-y834 (dataset.zip, folder dataset/original_data)
Split logic: By dive. Frames come from 312 videos, and one dive is often cut into several videos, so videos
    are grouped into dives with trashcan_dive_groups.json: videos that share a dive number or a date printed on
    the video overlay (read by OCR) are one group; a video whose overlay could not be read is its own group.
    The groups are shuffled (seed settings.random_state) and whole groups go to val until it holds about 20%
    of the frames. The official TrashCan train/val split is not used: 113 videos have frames in both.
Category mapping (from the original Supervisely tags, not the official COCO classes: the official animal_etc
    class mixes rays with squid, crabs, anemones and unknown objects):
  fish: animal tags fish, eel, ee; (typo), stingray (rays and a chimaera-like fish), 1 (eelpouts and grenadiers)
  non-fish: animal tags crab, 2 (spider crabs), starfish (sea stars and brittle stars), satrfish (typo),
    6 (a sea star), squid (squid and dumbo octopus), octopus (a large jellyfish), shrimp, crawfish (deep-sea
    shrimp and squat lobsters), lobster
  discard: animal tags shell / shells (empty clam shells, vent mussel beds, whelk shells), unknown (a crinoid,
    white blobs), jellyfish (faint blobs of doubtful size), turtle (a lump, not a turtle), 7 (red anemones),
    8 / 9 (a white object, likely trash), 3, 4, b; and the plant, rov, trash and unknown objects

7,212 frames (480x270 or 480x360, one per second of video) from ROV dives of JAMSTEC since 1982, mostly in the
Sea of Japan, selected for marine debris. Every object is a bitmap mask with a class (trash, rov, bio, unknown)
and tags; bio objects carry an "animal" or "plant" tag, and the animal tag names the animal type. The numeric
tags were checked by eye. Some trash and rov objects also carry an animal tag (3 fish, 1 crab); their class
follows the tag.
Boxes come from the masks. Mask parts under MIN_PART_AREA pixels (stray specks) are left out of the box. The
masks in MASK_FIXES cover several animals and get one box per part, or keep only the animal part. Boxes whose
long side is under MIN_BOX_SIDE pixels are dropped.
Videos in EXCLUDED_VIDEOS are left out: their boxes cover groups of animals and most animals have no box.
Animals are not boxed exhaustively: about 90% of fish have a box (fewer in shoals), but only about 1 in 4
non-fish (most brittle stars and sea stars, often whole fields of them, are not boxed). Frames without boxes
are kept.
"""

import base64
import hashlib
import json
import random
import re
import shutil
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "trashcan"
ZIP_URL = "https://conservancy.umn.edu/server/api/core/bitstreams/fffa78cc-3abf-42b9-b019-bc15a953a8e9/content"
ZIP_MD5 = "ef12cb7ec2fe1ae2f909c2345127bfe0"

# Every source category must be listed here explicitly; unlisted categories are discarded.
# Source categories are "animal: <animal tag>" for objects with an animal tag, else the object class.
CATEGORIES_FILTER = {
    "animal: fish": "fish",
    "animal: eel": "fish",
    "animal: ee;": "fish",  # typo of eel
    "animal: stingray": "fish",  # a ray half buried in sediment and a chimaera-like fish
    "animal: 1": "fish",  # eelpouts and grenadiers
    "animal: crab": "non-fish",
    "animal: 2": "non-fish",  # spider crabs
    "animal: starfish": "non-fish",  # sea stars and brittle stars
    "animal: satrfish": "non-fish",  # typo of starfish
    "animal: 6": "non-fish",  # a sea star
    "animal: squid": "non-fish",  # squid and dumbo octopus
    "animal: octopus": "non-fish",  # a large jellyfish with long tentacles
    "animal: shrimp": "non-fish",
    "animal: crawfish": "non-fish",  # deep-sea shrimp and squat lobsters
    "animal: lobster": "non-fish",
}
# Discarded: animal: shell, animal: shells (empty clam shells, vent mussel beds, whelk shells), animal: unknown
# (a crinoid, white blobs), animal: jellyfish (faint blobs of doubtful size), animal: turtle (a lump),
# animal: 7 (red anemones), animal: 8 and animal: 9 (a white object, likely trash), animal: 3 (unclear thin
# shapes), animal: 4 (trash), animal: b (a cluster of vent crabs), plant, rov, trash, unknown

DIVE_GROUPS_PATH = Path(__file__).with_name("trashcan_dive_groups.json")  # {video id: dive group label}
EXCLUDED_VIDEOS = {
    "000090",  # brittle-star bed: boxes cover tangles of several brittle stars, most are not boxed
    "000102",  # hydrothermal vent: dozens of crabs per frame, a few boxed, some boxes over clusters
}
# Masks (Supervisely object id) that cover several animals: "split" gives one box per part of at least
# 10% of the largest part, "largest" keeps the largest part only
MASK_FIXES = {
    411135292: "split",  # vid_000085_frame0000038: two eels
    411143306: "split",  # vid_000106_frame0000008: two sea stars
    411154148: "split",  # vid_000106_frame0000010: two sea stars
    411090802: "split",  # vid_000443_frame0000003: two crabs
    411135043: "largest",  # vid_000085_frame0000010: an eel; the other part is a sea star
}
MIN_PART_AREA = 10  # pixels
MIN_BOX_SIDE = 8  # pixels, long side
FILE_NAME = re.compile(r"^vid_(?P<video>\d{6})_frame(?P<frame>\d{7})\.jpg$")

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_data_dir / "dataset.zip"
source_dir = raw_data_dir / "dataset" / "original_data"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "trashcan_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name
preview_annotations_path = processing_dir / "trashcan_preview_coco.json"


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download dataset.zip from DRUM and extract the original frames and Supervisely annotations."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_marker = raw_data_dir / ".extracted"
    if extracted_marker.exists():
        print(f"TrashCan data already extracted at {source_dir}")
        return
    download_file_resumable(ZIP_URL, zip_path)
    actual = md5sum(zip_path)
    if actual != ZIP_MD5:
        raise ValueError(f"MD5 mismatch for {zip_path}: expected {ZIP_MD5}, got {actual}. Delete it and rerun.")
    print("Extracting dataset/original_data...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith("dataset/original_data/")]
        z.extractall(raw_data_dir, members)
    extracted_marker.touch()


def source_category(obj: dict) -> str:
    """'animal: <tag>' for an object with an animal tag, else its class ('plant' for plants)."""
    tags = {tag["name"]: str(tag.get("value")).strip() for tag in obj.get("tags", [])}
    if "animal" in tags:
        return f"animal: {tags['animal']}"
    if obj["classTitle"] == "bio" and "plant" in tags:
        return "plant"
    return obj["classTitle"]


def mask_boxes(obj: dict, width: int, height: int) -> list[list[float]]:
    """[x, y, w, h] boxes of a bitmap mask: one box around its parts, or per part for the masks in MASK_FIXES."""
    png = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
    mask = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    mask = (mask[:, :, 3] if mask.ndim == 3 else mask) > 0
    ox, oy = (int(v) for v in obj["bitmap"]["origin"])
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    parts = sorted(([ox + x, oy + y, ox + x + w, oy + y + h, area] for x, y, w, h, area in stats[1:].tolist()),
                   key=lambda part: -part[4])
    parts = [part for part in parts if part[4] >= MIN_PART_AREA] or parts[:1]
    fix = MASK_FIXES.get(obj["id"])
    if fix == "largest":
        parts = parts[:1]
    if fix == "split":
        groups = [[part] for part in parts if part[4] >= 0.1 * parts[0][4]]
    else:
        groups = [parts]
    boxes = []
    for group in groups:
        x0, y0 = max(0, min(p[0] for p in group)), max(0, min(p[1] for p in group))
        x1, y1 = min(width, max(p[2] for p in group)), min(height, max(p[3] for p in group))
        if max(x1 - x0, y1 - y0) >= MIN_BOX_SIDE:
            boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    return boxes


def convert_trashcan_to_coco():
    """One COCO file with every frame (copied with the dataset prefix) and a box per object, by source category."""
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    dive_groups = json.loads(DIVE_GROUPS_PATH.read_text())
    coco = {"images": [], "annotations": [], "categories": []}
    category_ids = {}
    images_path.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for annotation_path in sorted((source_dir / "annotations").glob("*.jpg.json")):
        name = annotation_path.name[: -len(".json")]
        parts = FILE_NAME.match(name)
        if parts["video"] in EXCLUDED_VIDEOS:
            stats["frames of excluded videos"] += 1
            continue
        data = json.loads(annotation_path.read_text())
        width, height = data["size"]["width"], data["size"]["height"]
        image_id = len(coco["images"]) + 1
        file_name = f"{DATASET_SHORTNAME}_{name}"
        shutil.copy2(source_dir / "images" / name, images_path / file_name)
        coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height,
                               "video": parts["video"], "frame": int(parts["frame"]),
                               "dive_group": dive_groups[parts["video"]]})
        for obj in data["objects"]:
            category = source_category(obj)
            if category not in category_ids:
                category_ids[category] = len(category_ids) + 1
                coco["categories"].append({"id": category_ids[category], "name": category})
            boxes = mask_boxes(obj, width, height)
            stats["masks split into several boxes"] += len(boxes) > 1
            stats["boxes dropped as too small"] += not boxes
            for bbox in boxes:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                            "category_id": category_ids[category], "bbox": bbox,
                                            "area": bbox[2] * bbox[3], "iscrowd": 0})

    print(f"Frames: {len(coco['images'])}, boxes: {len(coco['annotations'])}, {dict(stats)}")
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    partial.write_text(json.dumps(coco))
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_trashcan_to_coco()
    map_annotations_to_fish_and_non_fish(source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)


def write_preview_annotations():
    """Most frames have no box or one faint animal, so the preview samples frames with a fish and 2+ boxes."""
    with open(mapped_annotations_path) as f:
        coco = json.load(f)
    boxes, fish = Counter(), set()
    for annotation in coco["annotations"]:
        boxes[annotation["image_id"]] += 1
        if annotation["category_id"] == settings.fish_category_id:
            fish.add(annotation["image_id"])
    coco["images"] = [image for image in coco["images"] if image["id"] in fish and boxes[image["id"]] >= 2]
    preview_annotations_path.write_text(json.dumps(coco))


def validation_dive_groups(images: list[dict]) -> set[str]:
    """Whole dive groups, in seeded random order, until val holds about settings.train_val_split_ratio of the frames."""
    frames = Counter(image["dive_group"] for image in images)
    target = settings.train_val_split_ratio * len(images)
    groups = sorted(frames)
    random.Random(settings.random_state).shuffle(groups)
    val, count = set(), 0
    for group in groups:
        if count + frames[group] > 1.1 * target:
            continue
        val.add(group)
        count += frames[group]
        if count >= 0.95 * target:
            break
    return val


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        images = json.load(f)["images"]
    val_groups = validation_dive_groups(images)
    dive_group = {image["file_name"]: image["dive_group"] for image in images}
    should_the_image_be_included_in_train_set = lambda image_path: dive_group[Path(image_path).name] not in val_groups

    train_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    val_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
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
    write_preview_annotations()
    save_preview_image(images_path, preview_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
