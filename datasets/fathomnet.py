"""
FathomNet Dataset
Source: FathomNet API (image records and boxes per concept, images from their URLs)
Split logic: Random by image, reproducible (no location/camera grouping used)
Category mapping: every FathomNet concept is assigned to fish / non-fish / discard from its
    WoRMS lineage (see FISH_ROOTS, NON_FISH_ROOTS, DISCARD_ROOTS and MANUAL below). The
    resolved concept -> class mapping is committed in fathomnet_categories.json, so it does
    not change silently when FathomNet adds concepts; rebuild it with
    `python -m datasets.fathomnet --rebuild-categories`.
  fish: Actinopterygii, Sarcopterygii, Chondrichthyes, Agnatha + common names ("bony fish", ...)
  non-fish: motile animals > ~3 cm (tetrapods, sea stars, brittle stars, urchins, sea cucumbers,
    feather stars, decapods, cephalopods, large jellies, comb jellies, large snails, ...)
  discard: everything else (corals, sponges, anemones, sessile/small animals, tunicates, worms,
    siphonophores, "marine organism", ...)

Only images that contain at least one fish or non-fish box are downloaded. Boxes of
discarded concepts in those images are dropped.
Note: the download takes many hours. It can be interrupted and resumed: images already on
disk are skipped.
"""

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from fathomnet.api import boundingboxes, images, worms
from fathomnet.dto import GeoImageConstraints
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "fathomnet"
CATEGORIES_FILE = Path(__file__).with_name("fathomnet_categories.json")

# The deepest matching rank in a concept's WoRMS lineage decides its class
FISH_ROOTS = ["Actinopterygii", "Sarcopterygii", "Chondrichthyes", "Agnatha"]
NON_FISH_ROOTS = [
    "Tetrapoda",  # sea lions, dolphins, seals, turtles, seabirds
    "Asterozoa", "Echinoidea", "Holothuroidea", "Comatulida", "feather star",
    "Decapoda", "Stomatopoda", "Lophogastrida", "Aegidae", "Cirolanidae", "Colossendeidae",
    "Cephalopoda", "Neogastropoda", "Ringipleura", "Pseudothecosomata", "Pterotracheoidea",
    "Seguenziida", "Pleurotomariida",
    "Scyphozoa", "Staurozoa", "Cubozoa", "Narcomedusae", "Colobonema", "Halicreas",
    "Halitrephes", "Mitrocoma",
    "Lobata", "Beroida",
]
DISCARD_ROOTS = [  # override the non-fish roots above them
    "Myriapoda",  # homonym "Siphonophora"
    "Psolidae", "Isocrinida", "Hyocrinida", "Cyrtocrinida", "Crinoidea",  # attached sea cucumbers, stalked crinoids
    "Patellogastropoda", "Lepetellida", "Calyptraeidae",  # limpets
    "Trochida", "Provannidae", "Rubyspira", "Eulimidae", "Neritimorpha", "Pteropoda",  # small snails
    "Solmaris", "Solmundella",  # small medusae
    "Animalia", "Chordata", "Vertebrata", "Gnathostomata",  # generic parents ("Chordata" = blank frames)
]
# Labels with no WoRMS match even after the genus lookup, or with an ambiguous lineage
MANUAL = {
    "fish": [
        "bony fish", "shark", "ray or skate", "chimaera", "vertebrate", "Sebastomus complex",
        "HYPORTHODUSNIGRITUS-170021202", "Roundnose grenadier", "Leptocephalus-2", "Pleuronectiformes 2",
        "Rajiformes sp. A", "Macrourinae sp. 2", "Hoplostethus/Gephyroberyx", "Ijimaia/Ateleopus",
        "Bathytroctes/Narcetes", "Psednos/Paraliparis", "Synaphobranchus affinis/kaupii", "Barathrites/n. genus",
        "DiplacanthopomaA", "DiplacanthopomaB", "Bassogigas1", "Coelorinchus \"multibarred\"", "Iliophys cf. arx",
    ],
    "non-fish": [
        "brittle star", "urchin", "sea star", "sea cucumber", "Opiuroidea",
        "Asteroidea sp. 1", "Asteroidea sp. 3", "Asteroidea sp. 6", "Asteroidea sp. 7", "Asteroidea sp. 9",
        "Holothuroidea sp. 3", "Holothuroidea sp. 4", "Holothuroidea sp. 6", "Holothuroidea sp. 7",
        "Holothuroidea sp. 8", "Antedonoidea sp. 1", "Synallactidae gen. et sp. indet.", "Pedicellasteridae sp. 2",
        "shrimp", "crab", "Caridea sp. 2", "Polychelidae 2", "Cancridae sp. 1",
        "squid", "octopus", "cephalopod", "Teuthoidea", "Theudoidea", "Octopodinae",
        "Buccinidae sp. 1", "Neptunea-Buccinum Complex", "Aeolidiidae sp. 1", "sea turtle",
    ],
    "discard": [
        "Psolus sp.", "echinoderm", "gastropod", "Gastropoda sp. 2", "mollusc", "crustacean", "arthropod",
        "isopod", "sea spider", "sea butterfly", "chiton", "bivalve",
        "Holothuroidea/Actiniaria",  # sea cucumber or anemone
    ],
}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
raw_annotations_path = raw_data_dir / "dataset.json"
raw_images_path = raw_data_dir / "images"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
mapped_annotations_path = processing_dir / "fathomnet_mapped_annotations.json"
images_path = processing_dir / settings.images_folder_name


def get_ancestors_names(name: str) -> list[str]:
    """WoRMS lineage from "object" down to name; empty if WoRMS does not know the name."""
    for attempt in range(3):
        try:
            return worms.get_ancestors_names(name)
        except ValueError:  # 404: unknown name
            return []
        except Exception:
            if attempt == 2:
                raise
    return []


def get_genus(concept: str) -> str:
    """'Sebastolobus sp.' -> 'Sebastolobus', 'Peniagone vitrea- sp. 1 complex' -> 'Peniagone'"""
    name = re.split(r'[/\-"(]', concept)[0]
    name = re.split(r" (?:sp\.|cf\.|aff\.|nr\.|n\.|complex)| gen\.", name)[0]
    words = name.split()
    return words[0] if words else ""


def build_categories_filter() -> dict[str, str]:
    """Assign every FathomNet concept to "fish", "non-fish" or "discard"."""
    concepts = boundingboxes.find_concepts()
    manual = {c: target for target, names in MANUAL.items() for c in names}
    roots = {
        **{r: "fish" for r in FISH_ROOTS},
        **{r: "non-fish" for r in NON_FISH_ROOTS},
        **{r: "discard" for r in DISCARD_ROOTS},
    }

    def get_lineage(concept):
        lineage = get_ancestors_names(concept)
        if not lineage and get_genus(concept):
            lineage = get_ancestors_names(get_genus(concept))
        return concept, lineage

    with ThreadPoolExecutor(16) as executor:
        lineages = dict(executor.map(get_lineage, [c for c in concepts if c not in manual]))

    mapping = {}
    for concept in sorted(concepts):
        if concept in manual:
            mapping[concept] = manual[concept]
            continue
        # Check the concept name itself too: e.g. "hydroid" resolves to a single species
        names = [concept] + list(reversed(lineages[concept]))
        mapping[concept] = next((roots[n] for n in names if n in roots), "discard")
    return mapping


def load_categories_filter(rebuild: bool = False) -> dict[str, str]:
    if rebuild or not CATEGORIES_FILE.exists():
        mapping = build_categories_filter()
        with open(CATEGORIES_FILE, "w") as f:
            json.dump(mapping, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {len(mapping)} concepts to {CATEGORIES_FILE}")
    with open(CATEGORIES_FILE) as f:
        mapping = json.load(f)
    return {c: target for c, target in mapping.items() if target != "discard"}


CATEGORIES_FILTER = load_categories_filter(
    rebuild=__name__ == "__main__" and "--rebuild-categories" in sys.argv
)


def download_image(session: requests.Session, url: str, target: Path):
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(partial, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
                expected = r.headers.get("Content-Length")
            if expected is not None and partial.stat().st_size != int(expected):
                raise IOError(f"Incomplete download: {url}")
            partial.rename(target)
            return None
        except Exception as e:
            error = e
    partial.unlink(missing_ok=True)
    return f"{url}: {error}"


def fetch_concept_images(concept: str) -> list:
    """All FathomNet images with at least one box of concept (each with all its boxes)."""
    concept_images, offset = [], 0
    while True:
        for attempt in range(5):
            try:
                page = images.find(GeoImageConstraints(concept=concept, limit=1000, offset=offset))
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        if not page:
            return concept_images
        concept_images.extend(page)
        offset += len(page)


def generate_coco_annotations():
    """
    Write the COCO annotations of every image that has a fish or non-fish box. Boxes of
    discarded concepts are dropped. Replaces fathomnet-generate, which crashes on images
    whose width/height are missing in FathomNet; those are filled in after the download.
    """
    images_by_uuid = {}
    with ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(fetch_concept_images, c) for c in sorted(CATEGORIES_FILTER)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching image records"):
            for image in future.result():
                images_by_uuid[image.uuid] = image

    concepts = sorted(CATEGORIES_FILTER)
    concept_ids = {c: i for i, c in enumerate(concepts, start=1)}
    coco_images, coco_annotations = [], []
    for uuid in sorted(images_by_uuid):
        image = images_by_uuid[uuid]
        boxes = [b for b in image.boundingBoxes or [] if b.concept in concept_ids]
        if not boxes:
            continue
        image_id = len(coco_images) + 1
        coco_images.append({
            "id": image_id,
            "file_name": f"{image.uuid}.{image.url.split('.')[-1]}",
            "width": image.width,
            "height": image.height,
            "coco_url": image.url,
        })
        for box in boxes:
            coco_annotations.append({
                "id": len(coco_annotations) + 1,
                "image_id": image_id,
                "category_id": concept_ids[box.concept],
                "bbox": [box.x, box.y, box.width, box.height],
                "area": box.width * box.height,
                "iscrowd": 0,
            })

    coco_data = {
        "info": {"description": "FathomNet", "url": "https://fathomnet.org"},
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": c} for c, i in concept_ids.items()],
    }
    partial = raw_annotations_path.with_name(raw_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco_data, f)
    partial.rename(raw_annotations_path)
    print(f"Wrote {len(coco_images)} images and {len(coco_annotations)} boxes to {raw_annotations_path}")


def fill_missing_image_sizes():
    """Some FathomNet images have no width/height: read them from the downloaded files."""
    with open(raw_annotations_path) as f:
        coco_data = json.load(f)
    filled = 0
    for image in coco_data["images"]:
        path = raw_images_path / image["file_name"]
        if (image["width"] is None or image["height"] is None) and path.exists():
            with Image.open(path) as im:
                image["width"], image["height"] = im.size
            filled += 1
    if filled:
        with open(raw_annotations_path, "w") as f:
            json.dump(coco_data, f)
        print(f"Filled in the size of {filled} images")


def download_data():
    """
    Generate the COCO annotations of the fish and non-fish concepts from the FathomNet API,
    then download the images. Both steps skip what is already on disk.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_images_path.mkdir(parents=True, exist_ok=True)

    if raw_annotations_path.exists():
        print(f"Annotations already downloaded: {raw_annotations_path}")
    else:
        generate_coco_annotations()

    with open(raw_annotations_path) as f:
        coco_images = json.load(f)["images"]
    downloaded = set(os.listdir(raw_images_path))
    to_download = [
        (image["coco_url"], raw_images_path / image["file_name"])
        for image in coco_images
        if image["file_name"] not in downloaded
    ]
    print(f"Downloading {len(to_download)} of {len(coco_images)} images to {raw_images_path}")

    errors = []
    with requests.Session() as session, ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(download_image, session, url, target) for url, target in to_download]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result() is not None:
                errors.append(future.result())
    if errors:
        print(f"WARNING: {len(errors)} images could not be downloaded, e.g.:")
        for error in errors[:10]:
            print(f"  {error}")

    fill_missing_image_sizes()


def link_images_to_processing():
    """
    Hard-link the downloaded images into the processing folder (copy if linking fails), so
    that renaming them there leaves the raw folder intact for resuming the download.
    Images that could not be downloaded are removed from the annotations.
    """
    images_path.mkdir(parents=True, exist_ok=True)

    with open(mapped_annotations_path) as f:
        coco_data = json.load(f)

    # One directory listing each instead of a file check per image (slow on network drives)
    downloaded = set(os.listdir(raw_images_path))
    linked = set(os.listdir(images_path))
    kept_images = []
    for image in tqdm(coco_data["images"]):
        # On a rerun the annotations already hold the prefixed names
        file_name = remove_dataset_shortname_prefix_from_image_filename(
            Path(image["file_name"]).name, DATASET_SHORTNAME
        )
        if file_name not in downloaded:
            continue
        kept_images.append(image)
        if file_name in linked or f"{DATASET_SHORTNAME}_{file_name}" in linked:
            continue
        source = raw_images_path / file_name
        target = images_path / file_name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    missing = len(coco_data["images"]) - len(kept_images)
    if missing:
        print(f"WARNING: dropping {missing} images that are not on disk")
        kept_ids = {image["id"] for image in kept_images}
        coco_data["images"] = kept_images
        coco_data["annotations"] = [a for a in coco_data["annotations"] if a["image_id"] in kept_ids]
        with open(mapped_annotations_path, "w") as f:
            json.dump(coco_data, f, indent=2)


def clean_boxes():
    """
    A few FathomNet boxes have a zero or negative width/height (e.g. [504, 140, 47, -1]):
    drop them. Boxes that extend past the image border are clipped to it.
    """
    with open(mapped_annotations_path) as f:
        coco_data = json.load(f)
    image_sizes = {image["id"]: (image["width"], image["height"]) for image in coco_data["images"]}

    kept, dropped, clipped = [], 0, 0
    for annotation in coco_data["annotations"]:
        x, y, w, h = annotation["bbox"]
        width, height = image_sizes[annotation["image_id"]]
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            dropped += 1
            continue
        if [x1, y1, x2 - x1, y2 - y1] != [x, y, w, h]:
            annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            annotation["area"] = (x2 - x1) * (y2 - y1)
            clipped += 1
        kept.append(annotation)

    if dropped or clipped:
        print(f"Dropped {dropped} boxes with no area, clipped {clipped} boxes to the image")
        coco_data["annotations"] = kept
        # An image whose only boxes were dropped would become a false background image
        annotated_ids = {annotation["image_id"] for annotation in kept}
        empty = [image for image in coco_data["images"] if image["id"] not in annotated_ids]
        if empty:
            print(f"Dropped {len(empty)} images left without boxes")
            coco_data["images"] = [image for image in coco_data["images"] if image["id"] in annotated_ids]
        with open(mapped_annotations_path, "w") as f:
            json.dump(coco_data, f, indent=2)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    map_annotations_to_fish_and_non_fish(
        raw_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    clean_boxes()

    link_images_to_processing()

    add_dataset_shortname_prefix_to_image_names(
        images_path, mapped_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Random split by image, reproducible: sorted file names + fixed random_state
    with open(mapped_annotations_path) as f:
        image_names = sorted(Path(image["file_name"]).name for image in json.load(f)["images"])
    train_images, _ = train_test_split(
        image_names,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_images = set(train_images)

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
    # With --rebuild-categories, only fathomnet_categories.json is rebuilt (on import above)
    if "--rebuild-categories" not in sys.argv:
        main()
