"""
VIAME FishTrack23 — expert-annotated fish tracks in underwater videos and image sequences
from five collections: NOAA SEFSC SEAMAP reef fish video survey (Gulf of Mexico, greyscale),
NOAA PIFSC MOUSS (Hawaii deep-water bottomfish), Ifremer DropCam (baited cameras) and
Ifremer Game of Trawls (inside a pelagic trawl), both in the Bay of Biscay, and CDFW LakeCam
(Lake Jennings reservoir, California, freshwater)
Paper: Dawkins et al., "FishTrack23: An Ensemble Underwater Dataset for Multi-Object
Tracking", WACV 2024
Source: VIAME DIVE collection "FishTrack23"
(https://viame.kitware.com/#/collection/65a140e8a4c218785d408b42), Train and Test folders
(the 3 "Sample" sequences are not used). Every video is listed twice in the collection; the
"Video <name>.mp4" copies are unannotated duplicates and are skipped.
Split logic: The authors' Train → train and Test → val, by SEAMAP camera drop: the authors put
chunks of the same SEAMAP drop (and in a few cases the same recording) in both Train and
Test. Sequences are grouped into drops (same station number, e.g. 761901329, or the same
camera system on the same day starting within 30 minutes, e.g. SC2-Camera3-03-14-21-19-57);
a drop with any sequence in Train goes entirely to train (27 Test sequences, 25.6k boxes).
Category mapping:
  fish: every fish species, genus or family label (148 labels, see FISH_LABELS) and
    unspecified_fish
  non-fish: anomura (squat/hermit crabs, the size of the 10-20 cm reef fish around them),
    stenocionops_spinimana (a spider crab), non_fish_animal (a jellyfish, crabs, a crayfish)
  discard: non_fish_bait, non_fish_bait_apparatus, non_fish_plant

Only frames with at least one fish or non-fish box are kept. Frames are annotated at 5-30 Hz,
so consecutive frames of a sequence are near-duplicates; each image keeps its "sequence" and
"frame_index" so frames can be subsampled later.
"""

import csv
import io
import json
import re
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)


DATASET_SHORTNAME = "viame_fishtrack"
API_URL = "https://viame.kitware.com/api/v1"
COLLECTION_ID = "65a140e8a4c218785d408b42"
SOURCE_SPLITS = ["Train", "Test"]

FISH_LABELS = [
    # SEFSC SEAMAP
    "lutjanus_campechanus", "stenotomus_caprinus", "rhomboplites_aurorubens",
    "pagrus_pagrus", "epinephelus_morio", "ioglossus", "malacanthus_plumieri",
    "serranus_annularis", "balistes_capriscus", "mycteroperca_phenax", "calamus_nodosus",
    "lutjanus_griseus", "pristipomoides_aquilonaris", "calamus_leucosteus",
    "pristigenys_alta", "chaetodon_sedentarius", "serranus_phoebe",
    "holacanthus_bermudensis", "mycteroperca_microlepis", "haemulon_aurolineatum",
    "selene_vomer", "seriola_dumerili", "pterois", "halichoeres_bivittatus",
    "seriola_rivoliana", "centropristis_philadelphica", "opistognathus",
    "chaetodon_ocellatus", "caranx_crysos", "lachnolaimus_maximus", "serranus_subligarius",
    "lutjanus_synagris", "calamus_proridens", "holanthius_martinicensis",
    "canthigaster_rostratus", "caulolatilus_cyanops", "pomacentridae",
    "pomacanthus_arcuatus", "carcharhinus_falciformis", "lutjanus_buccanela", "chaetodon",
    "dermatolepis_inermis", "epinephelus_flavolimbatus", "calamus_bajonado",
    "seriola_zonata", "bodianus_rufus", "holocentrus_adscensionis", "seriola_fasciata",
    "halichoeres", "equetus_umbrosus", "serranus", "centropristis_ocyura",
    "mycteroperca_interstitialis", "mullidae", "diplectrum_formosum",
    "archosargus_probatocephalus", "bodianus_pulchellus", "gymnothorax_saxicola", "naso",
    "mycteroperca_bonaci", "upeneus_parvus", "decodon_puellaris", "ocyurus_chrysurus",
    "calamus", "paranthias_furcifer", "rypticus_maculatus", "lutjanus_vivanus",
    "urophycis_regia", "caulolatilus_chrysops", "gymnothorax_moringa", "sphyrna_lewini",
    "gobiidae", "serranus_atrobranchus", "haemulon_melanurum", "canthidermis_sufflamen",
    "carcharhinus", "synodontidae", "epinephelus_adscensionis", "balistes_vetula",
    "opistognathus_aurifrons", "chaetodipterus_faber", "caranx_ruber", "pomacanthus_paru",
    "muraena_retifera", "ophichthus_puncticeps", "anthiinae", "halichoeres_bathyphilus",
    "cephalopholis_cruentata", "mycteroperca", "liopropoma_eukrines",
    "caranx_bartholomaei", "syacium", "carcharhinus_plumbeus", "alectis_ciliaris",
    "holocentrus", "lutjanus_analis", "lutjanus", "epinephelus_nigritus", "holacanthus",
    "hypoplectrus", "pomacentrus_partitus", "decapterus", "chromis_enchrysurus",
    "carangidae", "caranx_melampygus", "acanthurus", "xanthichthys_ringens", "albula_vulpes",
    "sphyraena_barracuda", "rachycentron_canadum", "hypoplectrus_unicolor",
    "acanthurus_coeruleus", "lactophrys_trigonus", "chromis_insolata",
    "thalassoma_bifasciatum", "scomberomorus", "chromis_insolatus", "sparisoma_aurofrenatum",
    "pseudupeneus_maculatus", "halichoeres_garnoti", "scaridae", "priacanthus_arenatus",
    "epinephelus", "pomacanthus", "chromis", "gonioplectrus_hispanus", "equetus_lanceolatus",
    "diodontidae", "diplectrum", "haemulon_macrostomum", "scarus_vetula",
    "carcharhinus_perezi", "lutjanus_apodus", "chaetodon_aculeatus",
    # PIFSC MOUSS
    "etelis_coruscans", "aphareus_rutilans", "pristipomoides_zonatus",
    "pristipomoides_filamentosus", "seriola", "pristipomoides_sieboldii",
    "hyporthodus_quernus", "pristipomoides_auricilla", "etelis_carbunculus",
    # Ifremer DropCam and Game of Trawls
    "sparidae", "trachurus_trachurus", "sardina_pilchardus", "engraulis_encrasicolus",
    # CDFW LakeCam
    "micropterus_salmoides",
    # all collections
    "unspecified_fish",
]
CATEGORIES_FILTER = {
    **{label: "fish" for label in FISH_LABELS},
    "anomura": "non-fish",
    "stenocionops_spinimana": "non-fish",
    "non_fish_animal": "non-fish",
}
# Discarded: non_fish_bait, non_fish_bait_apparatus, non_fish_plant

# A SEAMAP drop is identified by its station number (in sequence or image file names) or by
# a camera recording (camera system, date, start time) such as SC2-Camera3-03-14-21-19-57
STATION_PATTERN = re.compile(r"(?<!\d)(7\d{8})(?!\d)")
RECORDING_PATTERN = re.compile(
    r"(SC\d)[-_]Camera\d[-_](\d\d-\d\d-\d\d)[-_](\d\d)-(\d\d)-\d\d"
)
SAME_DROP_MINUTES = 30

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
sequences_list_path = raw_download_path / "sequences.json"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
sequence_annotations_dir = processing_dir / "sequences"
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "viame_fishtrack_mapped_annotations.json"


def api_get(path: str) -> list | dict:
    response = requests.get(f"{API_URL}/{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def list_sequences() -> list[dict]:
    """
    List the annotated sequences (DIVE datasets) in the Train and Test folders of the
    collection, skipping the unannotated "Video <name>.mp4" copies.
    """
    if sequences_list_path.exists():
        with open(sequences_list_path) as f:
            return json.load(f)

    split_folders = api_get(
        f"folder?parentType=collection&parentId={COLLECTION_ID}&limit=0"
    )
    sequences = []
    for split_folder in split_folders:
        if split_folder["name"] not in SOURCE_SPLITS:
            continue
        for folder in api_get(
            f"folder?parentType=folder&parentId={split_folder['_id']}&limit=0"
        ):
            if folder["name"].startswith("Video "):
                continue
            assert folder["meta"].get("annotate"), f"Not a DIVE dataset: {folder['name']}"
            sequences.append(
                {
                    "name": folder["name"],
                    "id": folder["_id"],
                    "source_split": split_folder["name"],
                    "type": folder["meta"]["type"],
                }
            )

    sequences.sort(key=lambda s: s["name"])
    raw_download_path.mkdir(parents=True, exist_ok=True)
    with open(sequences_list_path, "w") as f:
        json.dump(sequences, f, indent=2)
    return sequences


def sequence_zip_path(sequence: dict) -> Path:
    return raw_download_path / sequence["source_split"] / f"{sequence['name']}.zip"


def download_sequence(sequence: dict, max_attempts: int = 5):
    """
    Download one sequence (media + annotations) as the zip that DIVE builds on the fly. The
    export has no Content-Length and ignores Range headers, so a broken download restarts.
    """
    zip_path = sequence_zip_path(sequence)
    if zip_path.exists():
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".part")
    url = f'{API_URL}/dive_dataset/export?folderIds=["{sequence["id"]}"]'

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(partial, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            with zipfile.ZipFile(partial) as zip_file:
                assert zip_file.testzip() is None
            partial.rename(zip_path)
            return
        except Exception as e:
            print(f"Download of {sequence['name']} failed (attempt {attempt}): {e}")
            time.sleep(30)
    raise RuntimeError(f"Could not download {sequence['name']}; rerun to retry")


def download_data():
    sequences = list_sequences()
    for i, sequence in enumerate(sequences, start=1):
        if not sequence_zip_path(sequence).exists():
            print(f"Downloading sequence {i}/{len(sequences)}: {sequence['name']}")
            download_sequence(sequence)
    print(f"All {len(sequences)} sequences downloaded to {raw_download_path}")


def read_viame_csv(zip_file: zipfile.ZipFile) -> tuple[float, list[list[str]]]:
    """Return the annotation frame rate and the detection rows of annotations.viame.csv."""
    [csv_name] = [n for n in zip_file.namelist() if n.endswith("annotations.viame.csv")]
    text = zip_file.read(csv_name).decode("utf-8")
    fps = None
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0].startswith("#"):
            for field in row:
                if field.startswith("fps: "):
                    fps = float(field.removeprefix("fps: "))
            continue
        rows.append(row)
    return fps, rows


def clip_box(row: list[str], width: int, height: int) -> list[float] | None:
    """Convert a VIAME TL/BR box to a COCO [x, y, w, h] box clipped to the image."""
    x1, y1, x2, y2 = (float(v) for v in row[3:7])
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def iterate_video_frames(video_path: Path, frame_indices: list[int], fps: float):
    """
    Yield (annotation frame index, image) for each annotated frame. DIVE places annotation
    frame f at time f / fps; the video is decoded sequentially and the native frame at that
    time is kept (annotated at 30 fps on a 29.97 fps video, two annotation frames can map to
    the same native frame).
    """
    capture = cv2.VideoCapture(str(video_path))
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    position = 0
    image = None
    for frame_index in frame_indices:
        native_index = round(frame_index * native_fps / fps)
        if native_index < position:
            if native_index == position - 1 and image is not None:
                yield frame_index, image
            continue
        while position < native_index:
            if not capture.grab():
                break
            position += 1
        ok, image = capture.read()
        if not ok:
            print(f"Frame {frame_index} not found in {video_path.name}")
            break
        position += 1
        yield frame_index, image
    capture.release()


def process_sequence(sequence: dict) -> Path:
    """
    Write the annotated frames of one sequence as JPEGs and save its images and boxes (with
    source category names) to sequences/<name>.json. Frames without any fish or non-fish box
    are skipped.
    """
    output_path = sequence_annotations_dir / f"{sequence['name']}.json"
    if output_path.exists():
        return output_path

    with zipfile.ZipFile(sequence_zip_path(sequence)) as zip_file:
        fps, rows = read_viame_csv(zip_file)
        rows = [row for row in rows if len(row) > 9 and row[9] in CATEGORIES_FILTER]

        # Annotation frame index -> rows; image sequences also need the image file name
        rows_by_frame = {}
        file_name_by_frame = {}
        for row in rows:
            frame_index = int(row[2])
            rows_by_frame.setdefault(frame_index, []).append(row)
            file_name_by_frame[frame_index] = row[1]
        frame_indices = sorted(rows_by_frame)

        if sequence["type"] == "video":
            [video_name] = [n for n in zip_file.namelist() if n.lower().endswith(".mp4")]
            with tempfile.TemporaryDirectory(dir=processing_dir) as temporary_dir:
                video_path = Path(temporary_dir) / Path(video_name).name
                with zip_file.open(video_name) as source, open(video_path, "wb") as target:
                    while chunk := source.read(16 * 1024 * 1024):
                        target.write(chunk)
                images, annotations = write_frames(
                    sequence, iterate_video_frames(video_path, frame_indices, fps), rows_by_frame
                )
        else:
            members = {Path(n).name: n for n in zip_file.namelist()}
            frames = (
                (
                    frame_index,
                    cv2.imdecode(
                        np.frombuffer(
                            zip_file.read(members[file_name_by_frame[frame_index]]), np.uint8
                        ),
                        cv2.IMREAD_COLOR,
                    ),
                )
                for frame_index in frame_indices
            )
            images, annotations = write_frames(sequence, frames, rows_by_frame)

    if len(images) < len(frame_indices):
        print(f"{sequence['name']}: {len(frame_indices) - len(images)} annotated frames missing")

    with open(output_path, "w") as f:
        json.dump({"images": images, "annotations": annotations}, f)
    return output_path


def write_frames(sequence: dict, frames, rows_by_frame: dict) -> tuple[list, list]:
    images = []
    annotations = []
    for frame_index, image in frames:
        height, width = image.shape[:2]
        file_name = f"{DATASET_SHORTNAME}_{sequence['name']}_{frame_index:06d}.jpg"
        boxes = [
            (row[9], box)
            for row in rows_by_frame[frame_index]
            if (box := clip_box(row, width, height)) is not None
        ]
        if not boxes:
            continue
        cv2.imwrite(str(coco_images_path / file_name), image)
        images.append(
            {
                "file_name": file_name,
                "height": height,
                "width": width,
                "sequence": sequence["name"],
                "source_split": sequence["source_split"],
                "frame_index": frame_index,
            }
        )
        annotations.extend(
            {"file_name": file_name, "category_name": name, "bbox": box} for name, box in boxes
        )
    return images, annotations


def build_coco_dataset(sequences: list[dict]):
    """Merge the per-sequence files into one COCO file with the source categories."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    category_names = sorted(CATEGORIES_FILTER)
    category_ids = {name: i for i, name in enumerate(category_names, start=1)}
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for n, i in category_ids.items()],
    }
    image_ids = {}
    for sequence in sequences:
        with open(sequence_annotations_dir / f"{sequence['name']}.json") as f:
            sequence_data = json.load(f)
        for image in sequence_data["images"]:
            image["id"] = image_ids[image["file_name"]] = len(image_ids) + 1
            coco_data["images"].append(image)
        for annotation in sequence_data["annotations"]:
            x, y, w, h = annotation["bbox"]
            coco_data["annotations"].append(
                {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_ids[annotation["file_name"]],
                    "category_id": category_ids[annotation["category_name"]],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_data, f)
    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")


def processing(max_workers: int = 8):
    sequences = list_sequences()
    coco_images_path.mkdir(parents=True, exist_ok=True)
    sequence_annotations_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sequence, s): s["name"] for s in sequences}
        for i, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"Processed {i}/{len(sequences)}: {futures[future]}")

    build_coco_dataset(sequences)

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def get_drop_keys(sequence: dict) -> set:
    """Station numbers and camera recordings in the sequence name and its image names."""
    names = [sequence["name"]]
    if sequence["type"] == "image-sequence":
        with zipfile.ZipFile(sequence_zip_path(sequence)) as zip_file:
            names += [row[1] for row in read_viame_csv(zip_file)[1]]
    keys = set()
    for name in names:
        keys.update(("station", m) for m in STATION_PATTERN.findall(name))
        for system, date, hour, minute in RECORDING_PATTERN.findall(name):
            keys.add(("recording", system, date, int(hour) * 60 + int(minute)))
    return keys


def get_val_sequences(sequences: list[dict]) -> set[str]:
    """
    Group SEAMAP sequences into camera drops and return the Test sequences whose whole drop
    is in Test. Two sequences are in the same drop if they share a station number, or a
    camera system and date with start times at most SAME_DROP_MINUTES apart.
    """
    parent = {s["name"]: s["name"] for s in sequences}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    keys = [
        (s["name"], key)
        for s in sequences
        if s["name"].startswith("SEFSC-SeaMap")
        for key in get_drop_keys(s)
    ]
    for i, (name_a, key_a) in enumerate(keys):
        for name_b, key_b in keys[i + 1 :]:
            same_station = key_a[0] == key_b[0] == "station" and key_a == key_b
            same_recording = (
                key_a[0] == key_b[0] == "recording"
                and key_a[1:3] == key_b[1:3]
                and abs(key_a[3] - key_b[3]) <= SAME_DROP_MINUTES
            )
            if same_station or same_recording:
                parent[find(name_a)] = find(name_b)

    train_drops = {find(s["name"]) for s in sequences if s["source_split"] == "Train"}
    val_sequences = {
        s["name"]
        for s in sequences
        if s["source_split"] == "Test" and find(s["name"]) not in train_drops
    }
    moved = sorted(
        s["name"]
        for s in sequences
        if s["source_split"] == "Test" and s["name"] not in val_sequences
    )
    print(f"{len(moved)} Test sequences share a drop with Train and go to train: {moved}")
    return val_sequences


def dataset_splitting():
    val_sequences = get_val_sequences(list_sequences())

    with open(mapped_annotations_path) as f:
        val_image_names = {
            image["file_name"]
            for image in json.load(f)["images"]
            if image["sequence"] in val_sequences
        }

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).name not in val_image_names
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
