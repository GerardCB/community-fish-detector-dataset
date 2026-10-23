# VLM pre-annotation of obsea_2013_2014

The [obsea_2013_2014](../../datasets/obsea_2013_2014.py) annotations (PANGAEA 946149) are incomplete:
in a hand count on 40 random photos about 60% of the visible fish have no box, mostly in schools,
and many human boxes are loose. Human annotators will review and clean the whole dataset; these
scripts pre-annotate it with Claude models first, to cut that work. Deleting a wrong box is quicker
than drawing a missing one, and a tightened box only needs a glance, so the pipeline favours adding
candidate boxes and marks how sure it is about each one.

The same pipeline was then used on [kso_mussel_bruv](../../datasets/kso_mussel_bruv.py) (see
[KSO Mussel Farm BRUV run](#kso-mussel-farm-bruv-run-2026-10-23)). The dataset is chosen with the
`VLM_DATASET` environment variable (default `obsea_2013_2014`); `pilot.DATASETS` holds each
dataset's paths, the scene description used in the prompts, and whether its human boxes have
fish / non-fish classes (OBSEA's are all fish).

## Pipeline

1. **Pass 1** (`pilot.py`, variant `claude-haiku-5-5_tiled_v2`): Claude Haiku 5.5 (effort medium)
   looks at each photo in 4 overlapping tiles (384x288 px, sent upscaled 3x) and returns a box, a
   class (fish / non-fish, following the dataset's annotation rules) and a confidence
   (high / medium / low) for every animal. Each tile keeps the boxes centred in its quadrant;
   duplicates are suppressed.
2. **Pass 2** (`second_pass.py`, reviewer `review_haiku_v2`): Claude Haiku 5.5 sees the photo
   (2x) with every box drawn and numbered, human boxes in green and pass-1 boxes in yellow, and
   the checklist of annotation problems (missing, extra, loose, two animals in one box, wrong
   class, unusable photo). For each box it answers keep, fix (new box and/or class) or remove,
   then lists the animals still missing. Photos with more than 20 boxes are reviewed in the same
   4 tiles (3x).
3. **Merge** (`second_pass.final_annotation`, policy `protect`): a box the review keeps stays
   exactly as it was; human boxes stay fish and are only removed on a high-confidence verdict
   (once in the 62 test photos, on what looks like a white filament); the review may remove a
   pass-1 box only if pass 1 gave it low confidence (see the 50-photo test below). Every removed
   box is listed in `rejected_candidates.json`.
4. **Export** (`export.py`): COCO for the annotators, see [Outputs](#outputs).

`batch_run.py` runs pass 1, pass 2 and the export on the Message Batches API at half price.
Requests carry their image inline, so they are split into batches of at most 200 MB (the API
limit is 256 MB); every stage is resumable and failed requests are resubmitted (3 attempts).

## Pilot results

All numbers below come from `pilot_results/`.

### Which model (9 hand-counted photos)

9 photos with 67 human boxes and 175 clearly visible fish without a box (`pilot9_images.tsv`).
"Existing found" = human boxes the model found again; "new" = model boxes on no human box, by
confidence. Full table: `pilot9_scores.json`; a hand check of new boxes: `pilot9_new_box_judgments.json`.

| variant | existing found | new high / medium / low | full dataset, Batch API* |
|---|---|---|---|
| Haiku 5.5, whole photo | 64% | 39 / 52 / 35 | $5 |
| Haiku 5.5, whole photo, effort max | 82% | 87 / 45 / 44 | $45 |
| **Haiku 5.5, 4 tiles, prompt v2 (pass 1)** | **82%** | **27 / 71 / 48** | **$17** |
| Sonnet 5.5, whole photo | 81% | 41 / 39 / 29 | $88 |
| Opus 5.5, whole photo | 93% | 50 / 89 / 38 | $246 |
| Opus 5.5, 4 tiles | 87% | 47 / 47 / 67 | $510 |

\* extrapolated from these 9 crowded photos, so higher than for typical photos (see the cost
analysis). Opus finds the most fish, but tiled Haiku gets most of the way at a fifteenth of the
cost. In a hand check, all of its high-confidence and most of its medium-confidence new boxes
were on fish, and about half of its low-confidence ones. Tiling helps more than effort: whole-photo
Haiku boxes in schools are often offset.

![The school photo of the pilot with every variant](pilot_results/figures/pilot9_school_all_variants.jpg)

### Both passes on 50 random photos

50 random photos (`random50_images.tsv`), judged one by one on before/after sheets
(`random50_judgments.json`): **39 better, 8 unchanged, 3 worse**. With the `protect` merge,
the 250 human boxes become 447 (`random50_export_summary.json`):

- 174 human boxes unchanged, 76 tightened by the review (about a third of them were loose)
- 177 boxes added by pass 1 (51 high, 71 medium, 55 low confidence), 20 added by the review
- 66 low-confidence pass-1 boxes removed by the review, kept aside as rejected candidates

![Before (human boxes) and after both passes](pilot_results/figures/random50_before_after_schools.jpg)

What went wrong, and what changed because of it:

- **The review removed real fish.** Of the 94 pass-1 boxes it removed, about 35 were clearly fish
  and 18 probably fish, against 12 correct removals (the rest unclear). The `protect` merge now
  lets it remove only low-confidence boxes, which are a coin flip either way.
  ![Pass-1 boxes the review removed](pilot_results/figures/review_removed_pass1_boxes.jpg)
- **A fish relabelled as non-fish** (a human-boxed damselfish): human boxes now always stay fish.
- **A big fish dropped by the merge**: pass 1 boxed it, but its box contained a small human box
  and was taken for a duplicate. A pass-1 box more than 4x the area of a human box inside it is
  now a different animal.
- **Still open:** low-confidence boxes on white drifting filaments in 3 of the 50 photos, and a
  few clearly visible fish that neither pass boxed (for example a large bream cut by the photo
  edge).

![Boxes on white filaments, the most common false box](pilot_results/figures/random50_before_after_filaments.jpg)

## Full run (2026-10-20)

Both passes on all 14,952 photos with the Batch API, in about 1.5 hours: 59,800 pass-1 and 21,389
pass-2 requests (91 of them asked again), **0 failed**. The 2 truncated photos were skipped.
Results are in `full_run_results/` (the export gzipped; `export.py` writes the same files to
`data/vlm_annotation/obsea_2013_2014/export/`).

| | boxes |
|---|---|
| human boxes, unchanged | 47,804 |
| human boxes, tightened by the review | 21,600 |
| added by pass 1 (high / medium / low confidence) | 65,846 (20,883 / 30,370 / 14,593) |
| added by the review (high / medium / low) | 5,778 (159 / 769 / 4,850) |
| **total** (fish / non-fish) | **141,028** (140,729 / 299) |
| left out: rejected candidates (pass-1 / human) | 24,853 (24,606 / 247) |

- The human boxes double: 69,651 become 141,028. 31% of the human boxes are tightened, and
  247 (0.35%) are removed on a high-confidence verdict (kept in `rejected_candidates.json`).
- 60 photos are marked unusable by the review (their human boxes are kept as they are).
- **Non-fish boxes are unreliable**: in a random sample of 18, most were fish (sea breams in
  schools, damselfish) or white filaments, and only one or two looked like a free-moving
  non-fish animal. There are only 299, and each counts towards `check_priority`.
- Review queue: 3,903 photos (26%) at check priority 0, 7,132 (48%) at 1-4, 3,917 (26%) at 5 or more.
- Measured cost: **$22.24** (pass 1 $12.32, pass 2 $9.93), $0.00149 per photo, as estimated.
  With the pilot ($2.43) the whole effort cost about $24.70.

## KSO Mussel Farm BRUV run (2026-10-23)

[KSO Mussel Farm BRUV](https://doi.org/10.5281/zenodo.14713926): 910 frames (640x360) of a GoPro
looking down at a black mat with bait at a Swedish mussel farm; the human boxes are gobies,
flatfish and crabs (`human_classes`: they keep their class). In a hand count on 64 frames
(`kso_mussel_bruv_results/hand_count_64_frames.tsv`) they miss 29% of the animals over ~3 cm.

- **Input**: `prepare_kso_mussel_bruv.py` writes the frames (Roboflow suffix removed) and the human
  boxes as fish / non-fish; the small snails (Gastropoda, discarded in the dataset) are left out,
  and the prompt says they and loose algae get no box.
- **Both passes** on all 910 frames: 3,640 + 910 requests, 0 failed, **$0.89**. An 18-frame pilot
  found 31 of 33 animals hand-counted as unboxed.
- **Cleaning** (`verify_added.py`): the 1,311 added boxes were filtered before going into the dataset.
  - `static`: 48 groups of boxes at the same spot in 3 or more frames of a recording, each
    checked by eye: 31 are objects (a curled algae frond and two fragments at the mat edge,
    pixel-identical for 28 minutes; debris; the mat clip; snails), 17 are animals that sat still
    or gobies overlapping by chance.
  - `submit` / `collect`: Claude Opus 5.5 checked every medium and low confidence box on a crop
    and the whole frame (993 requests, **$5.33**). It is too lenient on single frames (it called
    most boxes on the static objects "fish"), so it only decides the boxes the static check keeps.
  - every high-confidence box was checked by eye (about 95% correct; the errors were on the static
    objects, plus 1 class fix).
  - `select` with `kso_mussel_bruv_results/decisions.json` drops boxes on a human box (24; pass 1
    split some spider crabs in two), static objects (410) and the 17 Apr recordings (65, only
    snails), keeps the high-confidence boxes and the medium / low ones Opus confirmed with high or
    medium confidence, and writes `datasets/kso_mussel_bruv_added_boxes.json` (on the
    `gc/kso_mussel_bruv` branch): **572 boxes** (519 fish, 53 non-fish).
- **Result**: in a random check of 100 of the 572, about 94 are on a fish or crab and none has the
  wrong class. They recover about two thirds of the ~850 missing boxes estimated from the hand count.
- **Total cost $6.2** (pilot $0.02, both passes $0.89, Opus $5.33).

Results in `kso_mussel_bruv_results/`: the export (gzipped), the static groups, the Opus verdicts,
the decisions of the visual checks, the costs and the hand count.

```bash
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/prepare_kso_mussel_bruv.py
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/batch_run.py run
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/verify_added.py static
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/verify_added.py submit
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/verify_added.py collect
VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/verify_added.py select \
    --decisions tools/vlm_annotation/kso_mussel_bruv_results/decisions.json --out added_boxes.json
```

## Cost analysis

Prices (USD per million input / output tokens): Claude Haiku 5.5 $0.10 / $0.50, Sonnet 5.5
$2 / $10, Opus 5.5 $4 / $20; the Batch API halves them. Thinking tokens count as output.

| | per photo | 14,952 photos | with the Batch API |
|---|---|---|---|
| Pass 1 (4 tile requests) | $0.00162 | $24.3 | $12.1 |
| Pass 2 (1 request, 4 if crowded) | $0.00132 | $19.7 | $9.9 |
| **Both passes** | **$0.00294** | **$44.0** | **$22.0** |

- Measured on the 50 random photos with the standard API (`pilot.py` / `second_pass.py`), then
  multiplied by the 14,952 photos. A small Batch API run of both passes on 12 other photos
  (4 of them crowded) measured $0.0285 ($0.00237 per photo): all 75 requests succeeded, and
  each pass's batch finished in about 3-4 minutes. That sample was picked a third crowded, so it
  costs more per photo than the random 50 ($0.00147 per photo at Batch API prices).
- Expect roughly $18-28 for the full run: a crowded photo costs about 1.6x a sparse one in
  pass 1 and up to 4x in pass 2, and the share of crowded photos in the sample is uncertain.
- Pilot spend so far: $2.43 for the model comparison, the prompt and review iterations, the
  50-photo test and the batch test.
- The request images are uploaded with the batches: about 10 GB for pass 1 (59,808 tiles, about
  52 batches) and 5 GB for pass 2. Batches usually finish within an hour, at most 24 hours.
- For the annotators, the 50-photo test suggests: 17 of 50 photos at check priority 0 (only
  unchanged or tightened human boxes and high/medium-confidence model boxes), 20 at 1-4, 13 at 5
  or more.

## Running it

```bash
python3.11 -m venv venv-vlm
venv-vlm/bin/pip install -r tools/vlm_annotation/requirements.txt
# ANTHROPIC_API_KEY=... in .env (git-ignored); data/processing/obsea_2013_2014 must exist
venv-vlm/bin/python tools/vlm_annotation/batch_run.py run      # waits for the batches: use tmux or nohup
venv-vlm/bin/python tools/vlm_annotation/batch_run.py status   # from another shell
venv-vlm/bin/python tools/vlm_annotation/batch_run.py cost
```

`venv-vlm` is separate from the pipeline's `venv` because the Anthropic SDK needs a newer `idna`
than `roboflow` pins. `--images <tsv>` limits a run to some photos and `--run-dir` keeps runs
apart. A stopped run continues where it left off when started again.

On its first start a run copies the photo list (with ids) and the human boxes into its run dir
(`photos.json`, `human_boxes.json`) and reads the photos from the raw download, so rebuilding
`data/processing/obsea_2013_2014` during a run changes nothing. It also skips photos whose file
cannot be decoded completely: 2 photos of the PANGAEA zip are truncated (most of the picture is
missing).

## Outputs

In `data/vlm_annotation/obsea_2013_2014/export/` (git-ignored, like all data):

- `obsea_2013_2014_vlm_coco.json`: every photo and its final boxes, categories fish = 1 and
  non-fish = 2. Each annotation has `attributes`: `source` (human, vlm_pass1, vlm_review),
  `change` (unchanged, box_corrected, added), `confidence` (model boxes) and `original_bbox`
  (tightened human boxes). Each image has `vlm`: status, unusable, crowded, check_priority, counts.
- `rejected_candidates.json`: the boxes the review removed, per photo (low-confidence pass-1
  boxes, and the rare human box removed on a high-confidence verdict).
- `review_queue.csv`: photos sorted by `check_priority`, highest first.
- `summary.json`, and `../cost.json` with the measured tokens and cost.

Image ids follow the numbering of the original 14,952-photo `obsea_2013_2014` COCO; the dataset
script now skips the 2 truncated photos, so join with it on `file_name`. Those 2 photos are in the
export with status `damaged_file` and should be dropped.

`check_priority` counts the uncertain boxes of a photo (low-confidence model boxes,
medium-confidence boxes added by the review, rejected candidates), plus 3 if the photo is crowded
and 5 if it is unusable or could not be reviewed.

## Files

- `pilot.py`: pass 1 and the model/variant comparison (`run`, `score`, `render`, `review`)
- `second_pass.py`: pass 2, the merge policies and the before/after sheets (`run`, `render`, `final`)
- `batch_run.py`: both passes on the Batch API
- `export.py`: COCO export for the annotators
- `prepare_kso_mussel_bruv.py`: the KSO frames and human boxes as pipeline input
- `verify_added.py`: static-object groups, the Opus check and the selection of added boxes (KSO)
- `kso_mussel_bruv_results/`: results and decisions of the KSO run
- `pilot_results/`: photo lists, hand counts, scores, judgments and figures of the pilot
