# Dataset Processing Details

Each dataset script in `datasets/` follows a 4-step pattern: Download -> Process -> Preview -> Split.

## Complete Datasets

### brackish
- **Source**: [Roboflow](https://public.roboflow.com/ds/vGBLxigwno?key=bhFPGoB3VB)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one)
- **Category mapping**:
  - fish: `fish`, `small_fish`
  - non-fish: `crab`, `starfish`, `jellyfish`, `shrimp`
  - discard: `animals` (Roboflow supercategory, no annotations)
- **Split**: By deployment site (extracted from image filename prefix before `_jpg`)

### deep_vision
- **Source**: [NMDC FTP](https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip)
- **Download**: Automatic
- **Annotations**: CSV -> COCO conversion
- **Category filter**: None (all are fish)
- **Split**: By deployment (first token of filename before `_`)

### deepfish
- **Source**: [QLD Data](http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar)
- **Download**: Automatic
- **Annotations**: Segmentation masks -> bounding boxes (connected component analysis). Uses the 310 `valid` frames (with fish) and the 310 `empty` frames (no fish, kept as background)
- **Category mapping**:
  - fish: `fish` (every mask component of at least 100 px)
  - non-fish: none
  - discard: mask noise (14 components of 1-20 px)
- **Split**: By habitat (first token of filename before `_`, 19 habitats). Habitats with fish (13) and habitats with only empty frames (6) are split separately so val is not mostly empty

### fathomnet
- **Source**: [FathomNet API](https://fathomnet.org/)
- **Download**: Automatic via `fathomnet-generate` CLI (~12 hours)
- **Annotations**: COCO format from FathomNet API (fish concepts: Actinopterygii, Sarcopterygii, Chondrichthyes, Myxini descendants)
- **Category filter**: None (filtered at download)
- **Split**: Random

### fishclef
- **Source**: [Zenodo](https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1)
- **Download**: Automatic
- **Annotations**: XML -> COCO conversion, frames extracted from .flv videos
- **Category filter**: None (majority "Null" = general fish)
- **Split**: By video ID

### mit_river_herring
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip)
- **Download**: Automatic
- **Annotations**: COCO format (multiple sub-datasets joined into one)
- **Category filter**: None (only herring)
- **Split**: By location+video (first two tokens of filename)

### noaa_puget
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/noaa-psnf/)
- **Download**: Automatic (separate image and annotation downloads)
- **Annotations**: COCO format (images containing `fish_or_crab` / `unknown` boxes are dropped entirely)
- **Category mapping**:
  - fish: `fish`
  - non-fish: `crab`
  - discard: `empty` (no-animal marker, no bbox), `fish_or_crab`, `unknown`
- **Split**: By camera (`SDxx` camera ID, first token of filename)

### project_natick
- **Source**: [GitHub](https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip)
- **Download**: Automatic
- **Annotations**: Pascal VOC -> COCO conversion. Both subsets are used: `fish_arrow_worms_annotation` (1,072 frames, one video) and `fish_annotations` (46 frames, two videos)
- **Category mapping**:
  - fish: `Fish`
  - non-fish: none
  - discard: `Squid`. Despite the label these are arrow worms (chaetognaths), thin translucent planktonic worms mostly under ~3 cm
- **Split**: By chunks of 25 consecutive frames per video (49 chunks). There are only 3 videos, one with 96% of the frames, so a per-video split is not possible

### roboflow_fish
- **Source**: [Roboflow](https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one). The export holds 670 of the 680 source images twice (pixel-identical, identical boxes, one copy with Latin family names and one with common names); one copy is kept, by source filename (the part before `.rf.<hash>`)
- **Category mapping**:
  - fish: all 26 source classes (Latin family names, e.g. `Scombridae -Tunas-`, and common names, e.g. `tuna`) plus Roboflow's parent category `fish`
  - non-fish: none
  - discard: none
- **Split**: Pre-provided (uses Roboflow's original split)

### viame_fishtrack
- **Source**: [VIAME](https://viame.kitware.com/)
- **Download**: Automatic (separate train/val URLs)
- **Annotations**: VIAME CSV -> COCO conversion, frames extracted from videos
- **Category mapping**:
  - fish: `micropterus_salmoides`, `unspecified_fish`, `etelis_coruscans`, `hyporthodus_quernus`, `pristipomoides_zonatus`, `pristipomoides_auricilla`, `seriola`, `caranx_melampygus`
  - non-fish: `non_fish_animal`
  - discard: `non_fish_bait`, `non_fish_plant`
- **Split**: Pre-split by URL (train and val downloaded separately)

### zebrafish
- **Source**: [Kaggle](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)
- **Download**: Automatic via kagglehub
- **Annotations**: CSV -> DataFrame -> COCO conversion (2,224 frames, 3 boxes per frame)
- **Category mapping**:
  - fish: `Zebrafish` (6,672 boxes, including those flagged as occluded or glitched)
  - non-fish: none
  - discard: none
- **Split**: By video (Vid1 -> train, Vid2 -> val, hardcoded). Only two videos of the same tank, so any finer split would put near-identical frames in both sets

### orange_chromide
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/7w45jx35hd/1)
- **Download**: Automatic
- **Annotations**: YOLO TXT -> COCO conversion (single class, 640x640 images)
- **Category filter**: None (single class — fish)
- **Split**: Random (all images from same pond environment, no location metadata)

### pomerfish
- **Source**: [Zenodo](https://zenodo.org/records/17432128)
- **Download**: Automatic (`PomerFish.rar`, 25.1 GB via `unar`)
- **Annotations**: COCO format (already provided, 1-indexed)
- **Category filter**: None (all 10 categories are freshwater fish species)
- **Split**: By video/deployment ID (extracted from `dataset_<id>_frame_<n>.PNG` filename pattern, 27 unique IDs)

## Partial Datasets

These datasets require manual download or have special dependencies.

### coralscapes
- **Source**: [Coralscapes](https://josauder.github.io/coralscapes/)
- **Download**: Manual - download Parquet files and place in `data/raw/coralscapes/`
- **Annotations**: Parquet -> segmentation masks -> bounding boxes (connected components)
- **Category filter**: fish only (category ID 9 in source, remapped to 1)
- **Split**: By site (sites 1-24 train, sites 25-35 val)
- **Special**: Requires `pyarrow` for Parquet reading

### torsi
- **Source**: [CSIRO](https://data.csiro.au/collection/64913)
- **Download**: Manual - download zip and place in `data/raw/torsi/`
- **Annotations**: COCO format (already provided)
- **Category filter**: `orange_roughy`, `eel`, `misc_fish`, `orange_roughy_edge`, `chimera`, `oreo`, `shark`, `whiptail`
- **Split**: By date (2019-07-13/14/15 train, 2019-07-16/17 val, ~19% val ratio)

### f4k
- **Source**: [UniCT OneDrive](https://bit.ly/f4k-detection-tracking)
- **Download**: Manual - download `f4k_detection_tracking.zip` and place in `data/raw/f4k/`
- **Annotations**: XML contour annotations -> bounding boxes, keyframes extracted from `gt_<video_id>.flv` videos with OpenCV
- **Category mapping**:
  - fish: `fish`
  - non-fish: none
  - discard: `open_sea`, `sea`, `rocks`, `coral`, `plant`, `dark_area`, `other`, `algae` (background regions, one polygon per scene)
- **Split**: By video ID (106-109 val, 110-124 train)

### kakadu
- **Source**: Manual download
- **Download**: Manual - place dataset in `data/raw/kakadu/`
- **Annotations**: COCO format (cleaned to remove empty bboxes)
- **Category filter**: None (all are fish)
- **Split**: Random

### marine_detect
- **Source**: Roboflow (two datasets: FishInv + Megafauna)
- **Download**: Manual - download both datasets from Roboflow and place in `data/raw/marine_detect/`
- **Annotations**: YOLO text -> COCO conversion, two datasets merged
- **Category filter**: `turtle`, `ray`, `shark`, `bolbometopon_muricatum`, `chaetodontidae`, `cheilinus_undulatus`, `cromileptes_altivelis`, `fish`, `haemulidae`, `lutjanidae`, `muraenidae`, `scaridae`, `serranidae`
- **Split**: By original split suffix in filename (train/valid/test)
- **Special**: OzFish images are skipped to avoid duplication
