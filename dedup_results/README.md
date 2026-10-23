# Duplicate check: first run on 7 datasets

Results of [`datasets/deduplicate.py`](../datasets/deduplicate.py) on the datasets processed locally. The check is analytical: nothing was removed from the datasets.

## Setup

- **Datasets:** `digi4eco`, `kso_mussel_bruv`, `obsea_2013_2014`, `ruod_r`, `trashcan`, `uiis10k`, `yolo_bruv` (54,133 images in 14 train/val splits).
- **Method:** each image gets an [SSCD](https://github.com/facebookresearch/sscd-copy-detection) copy-detection descriptor; similarity is the cosine of two descriptors. Files with the same SHA-1 are exact copies.
- **Duplicate:** the same photo in two datasets, or in the train and val splits of one dataset, with similarity >= 0.9.
- **Near-identical frames:** pairs inside one split of one dataset with similarity >= 0.995. They are not duplicates: as agreed with Filippo, they stay in the dataset and are listed so training can subsample them.

## Calibration

| Test | Result |
|---|---|
| Best match between unrelated datasets (other places and years) | never above 0.40 |
| Neighbouring video frames, 1 s apart | 0.85-0.9 |
| Frames of a fixed camera | often above 0.95 |
| Same OBSEA scene on different days | up to 0.87 |
| Inside one split, >= 0.999 | the same frame repeated by the video, pixel-identical |
| Inside one split, 0.99-0.997 | consecutive frames with fish moved a little |

600 synthetic copies of random images, share found at each threshold:

| Copy | >= 0.8 | >= 0.85 | >= 0.9 |
|---|---|---|---|
| resized to half, re-encoded at JPEG quality 50 | 90% | 79% | 61% |
| stretched to 640x640 | 100% | 100% | 100% |
| cropped up to 10% per side | 100% | 93% | 62% |
| text overlay | 100% | 99% | 71% |
| crop + stretch + black text banner | 44% | 19% | 1% |
| horizontal flip | 100% | 100% | 100% |

At 0.9, heavily edited copies can be missed, but neighbouring video frames are not mistaken for copies.

## Findings

**No byte-identical files**, in any dataset or split.

**Between datasets:**
- **UIIS10K holds 161 TrashCan frames** (128 in UIIS10K train, 33 in val), all at >= 0.95. Other frames of the same TrashCan videos also match at 0.85-0.9, but those are neighbouring frames, not copies.
- **RUOD and UIIS10K share 6 photos** at >= 0.9. Their other matches, at 0.75-0.85, are different frames of the same dive footage.

**Between train and val inside a dataset** (val images with a train image at >= 0.9):

| Dataset | Val images | >= 0.9 | >= 0.95 | What they are |
|---|---|---|---|---|
| RUOD | 2,624 | 59 | 7 | the same photo in both splits |
| UIIS10K | 1,907 | 19 | 0 | the same photo in both splits |
| OBSEA | 3,242 | 4 | 0 | the same fixed-camera scene, not copies |
| YOLO-BRUV, TrashCan, KSO, DIGI4ECO | | 0 | 0 | |

**Drop estimate:** keeping one image per duplicate group (a val image if there is one, else the one with most boxes) would drop **245 of 54,133 images**: UIIS10K 147, RUOD 58, TrashCan 38, OBSEA 2.

**Near-identical frames kept for subsampling** (690 pairs):

| Dataset | Pairs | Note |
|---|---|---|
| DIGI4ECO | 635 (165 train, 470 val) | frames the videos repeat |
| RUOD | 43 | includes the same photo twice with different boxes (e.g. `ruod_r_005047` and `ruod_r_006380`, 37 and 48 boxes) |
| TrashCan | 10 | |
| KSO | 2 | |

## Files

- [`report.md`](report.md): the full report of the run.
- [`clusters.json`](clusters.json): the 245 duplicate groups, each a list of `<dataset>_<split>/<file name>`.
- [`near_identical_frames.csv`](near_identical_frames.csv): the 690 near-identical pairs, with similarity, dataset, split, both file names and box counts.

`pairs.csv` (every candidate pair, 35 MB) is not included. Run the script to regenerate it:

```
pip install -r requirements.txt -r requirements-dedup.txt
python -m datasets.deduplicate
```
