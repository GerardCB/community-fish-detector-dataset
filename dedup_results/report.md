# Duplicate report

54,133 images in 7 datasets. A duplicate is a pair of images in two datasets or in the two splits of one dataset with SSCD cosine similarity >= 0.9, or identical files (SHA-1). Pairs inside one split from 0.995 up are near-identical frames: they stay, and are listed for subsampling in training.

## Images with a match in another dataset, by similarity of their best match

Each row counts the images of the first dataset whose most similar image in the second one reaches the similarity. Neighbouring frames of a video also match at 0.75-0.9, identical photos mostly above 0.9.

| dataset | split | matched in | >= 0.70 | >= 0.75 | >= 0.80 | >= 0.90 | >= 0.95 |
|---|---|---|---|---|---|---|---|
| ruod_r | train | uiis10k (train, val) | 101 | 68 | 33 | 4 | 1 |
| ruod_r | val | uiis10k (train, val) | 36 | 22 | 13 | 2 | 0 |
| trashcan | train | uiis10k (train, val) | 2,072 | 1,553 | 1,116 | 531 | 248 |
| trashcan | val | uiis10k (train, val) | 228 | 166 | 135 | 67 | 40 |
| uiis10k | train | ruod_r (train, val) | 75 | 51 | 30 | 6 | 1 |
| uiis10k | train | trashcan (train, val) | 140 | 133 | 129 | 128 | 128 |
| uiis10k | val | ruod_r (train, val) | 15 | 12 | 7 | 0 | 0 |
| uiis10k | val | trashcan (train, val) | 37 | 34 | 33 | 33 | 33 |

## Val images with a match in the train split of their own dataset

| dataset | val images | >= 0.70 | >= 0.75 | >= 0.80 | >= 0.90 | >= 0.95 |
|---|---|---|---|---|---|---|
| obsea_2013_2014 | 3,242 | 1,735 | 978 | 227 | 4 | 0 |
| ruod_r | 2,624 | 589 | 474 | 327 | 59 | 7 |
| trashcan | 1,408 | 317 | 113 | 20 | 0 | 0 |
| uiis10k | 1,907 | 311 | 190 | 120 | 19 | 0 |
| yolo_bruv | 574 | 394 | 235 | 77 | 0 | 0 |

In datasets from fixed cameras or videos, these are mostly frames of the same scene, not copies.

## Near-identical frames inside one split of a dataset (kept, listed in near_identical_frames.csv)

Pairs from 0.995 up are frames repeated by a video or with fish moved by a few pixels; below it (from 0.9) they are mostly consecutive frames of a video or a fixed camera.

| dataset | split | pairs >= 0.9 | pairs >= 0.995 | images in them | identical files |
|---|---|---|---|---|---|
| digi4eco | train | 23,557 | 165 | 135 | 0 |
| digi4eco | val | 6,059 | 470 | 665 | 0 |
| kso_mussel_bruv | train | 2,798 | 2 | 4 | 0 |
| kso_mussel_bruv | val | 730 | 0 | 0 | 0 |
| obsea_2013_2014 | train | 984 | 0 | 0 | 0 |
| obsea_2013_2014 | val | 487 | 0 | 0 | 0 |
| ruod_r | train | 3,083 | 31 | 61 | 0 |
| ruod_r | val | 279 | 12 | 15 | 0 |
| trashcan | train | 10,521 | 10 | 20 | 0 |
| trashcan | val | 2,958 | 0 | 0 | 0 |
| uiis10k | train | 47 | 0 | 0 | 0 |
| uiis10k | val | 2 | 0 | 0 | 0 |
| yolo_bruv | train | 9,132 | 0 | 0 | 0 |
| yolo_bruv | val | 1,624 | 0 | 0 | 0 |

## If one image per duplicate group were kept

245 groups, linked by cross-dataset and train-val pairs at >= 0.9 where each image is the other's best match, and by identical files. Keeping one image per group (a val image if there is one, else the one with most boxes) would drop:

| dataset | train images dropped | val images dropped | share of its images |
|---|---|---|---|
| obsea_2013_2014 | 2 | 0 | 0.0% |
| ruod_r | 58 | 0 | 0.4% |
| trashcan | 37 | 1 | 0.5% |
| uiis10k | 144 | 3 | 1.5% |

Total: 245 of 54,133 images.
