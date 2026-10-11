# Example: fathomnet (API-based download + taxonomy-based mapping + random split)

This is an example of a dataset that:
- Uses an API/CLI for downloading (not a direct URL), resumable: files already on disk are skipped
- Has thousands of source classes (taxonomic concepts), so `CATEGORIES_FILTER` is derived from the WoRMS lineage of each concept with root rules, and the resolved mapping is committed as a JSON file next to the script (`datasets/fathomnet_categories.json`) so it does not change silently between runs
- Only downloads images that contain a fish or non-fish box (the concept list is passed to the CLI)
- Already has COCO-format annotations
- Uses a reproducible random split (sorted file names + `random_state`), because no location/deployment grouping is used

See `datasets/fathomnet.py` for the full script. Abridged:

```python
DATASET_SHORTNAME = "fathomnet"
CATEGORIES_FILE = Path(__file__).with_name("fathomnet_categories.json")

# The deepest matching rank in a concept's WoRMS lineage decides its class
FISH_ROOTS = ["Actinopterygii", "Sarcopterygii", "Chondrichthyes", "Agnatha"]
NON_FISH_ROOTS = ["Tetrapoda", "Asterozoa", "Echinoidea", "Decapoda", "Cephalopoda", ...]
DISCARD_ROOTS = ["Psolidae", "Crinoidea", ..., "Animalia", "Chordata"]  # override the roots above them
MANUAL = {"fish": ["bony fish", "shark", ...], "non-fish": ["brittle star", "urchin", ...], "discard": [...]}


def build_categories_filter() -> dict[str, str]:
    """Assign every FathomNet concept to "fish", "non-fish" or "discard"."""
    # For each concept in boundingboxes.find_concepts():
    #   - MANUAL wins
    #   - otherwise walk [concept] + reversed(worms.get_ancestors_names(concept)) and stop at
    #     the first root; if WoRMS does not know the concept, look up its genus ("X sp." -> "X")
    #   - nothing matched -> discard
    # Note: worms.get_descendants_names(root) is NOT used: it misses the common names that
    # FathomNet links to a taxon (e.g. "feather star").
    ...


def load_categories_filter(rebuild: bool = False) -> dict[str, str]:
    if rebuild or not CATEGORIES_FILE.exists():
        ...  # build_categories_filter() and write CATEGORIES_FILE
    with open(CATEGORIES_FILE) as f:
        mapping = json.load(f)
    return {c: target for c, target in mapping.items() if target != "discard"}


CATEGORIES_FILTER = load_categories_filter(
    rebuild=__name__ == "__main__" and "--rebuild-categories" in sys.argv
)


def download_data():
    # 1. If dataset.json is missing: fetch the image records of every concept in
    #    CATEGORIES_FILTER with fathomnet.api.images.find (paged, in parallel), keep only the
    #    boxes of those concepts, and write the COCO file (via .part + rename).
    #    fathomnet-generate is not used: it crashes on images whose width/height are missing
    #    in FathomNet, and its --img-download writes dataset.json before fetching any image.
    # 2. Download every image in dataset.json from its coco_url that is not on disk yet:
    #    check the HTTP status, write to <name>.part and rename when complete.
    # 3. Fill in missing width/height from the downloaded files.
    ...


def processing():
    map_annotations_to_fish_and_non_fish(raw_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)
    # Hard-link the raw images into processing/JPEGImages (keeps the raw folder intact for
    # resuming) and drop images that failed to download from the annotations
    link_images_to_processing()
    add_dataset_shortname_prefix_to_image_names(images_path, mapped_annotations_path, DATASET_SHORTNAME)


def dataset_splitting():
    # Random split by image, reproducible: sorted file names + fixed random_state
    with open(mapped_annotations_path) as f:
        image_names = sorted(Path(image["file_name"]).name for image in json.load(f)["images"])
    train_images, _ = train_test_split(
        image_names, test_size=settings.train_val_split_ratio, random_state=settings.random_state
    )
    train_images = set(train_images)
    ...  # split_coco_dataset_into_train_validation(..., lambda p: Path(p).name in train_images)


def main():
    download_data()                                                                 # 1. DOWNLOAD
    processing()                                                                    # 2. PROCESS
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)    # 3. PREVIEW
    dataset_splitting()                                                             # 4. SPLIT


if __name__ == "__main__":
    # With --rebuild-categories, only fathomnet_categories.json is rebuilt (on import above)
    if "--rebuild-categories" not in sys.argv:
        main()
```
