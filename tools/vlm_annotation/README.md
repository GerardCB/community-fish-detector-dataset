# Claude pre-annotation of incomplete datasets

Many datasets box only some of the animals in their photos. These scripts ask Claude models to propose the
missing boxes, review the dataset's boxes, and check the added ones, so that a person only has to accept or
reject them. The [`/pre-annotate`](../../.claude/skills/pre-annotate/SKILL.md) Claude Code skill walks
through a whole run: it runs a pilot first, estimates the cost of the whole dataset for each option (pass 1
only, with the review, with the Opus check) and asks before running anything at full scale.

The pipeline was built on OBSEA 2013-2014 (fixed camera, about 60% of the fish unboxed) and then used on
KSO Mussel Farm BRUV. The pilots, figures and full results of those runs are on the
[`gc/vlm-fish-annotation`](https://github.com/GerardCB/community-fish-detector-dataset/tree/gc/vlm-fish-annotation/tools/vlm_annotation)
branch.

## How it works

1. **Pass 1** (`pilot.py`, variant `claude-haiku-5-5_tiled_v2`): Claude Haiku 5.5 looks at each photo in 4
   overlapping tiles (each 60% of the photo's width and height, enlarged) and returns a box, a class (fish /
   non-fish, by the rules of CLAUDE.md) and a confidence (high / medium / low) for every animal. Each tile keeps
   the boxes centred in its quadrant; duplicates are suppressed.
2. **Pass 2** (`second_pass.py`, reviewer `review_haiku_v2`): Claude Haiku 5.5 sees the photo with every box
   drawn and numbered (human boxes in green, pass-1 boxes in yellow) and the checklist of annotation problems.
   For each box it answers keep, fix (new box and/or class) or remove, then lists the animals still missing.
   Photos with more than 20 boxes are reviewed in the same 4 tiles.
3. **Merge** (`second_pass.final_annotation`, policy `protect`): a box the review keeps stays as it was; human
   boxes keep their class and are only removed on a high-confidence verdict; the review may remove a pass-1
   box only if pass 1 gave it low confidence. Removed boxes are kept in `rejected_candidates.json`.
4. **Export** (`export.py`): a COCO file in which every box has `attributes`: `source` (`human`, `vlm_pass1`,
   `vlm_review`), `change` (`unchanged`, `box_corrected`, `added`) and `confidence`; plus a review queue of
   the photos that most need a person.
5. **Cleaning** (`verify_added.py`), before added boxes go into a dataset: groups of boxes at the same place
   in several frames of a video (static objects) are checked by eye, medium and low confidence boxes are
   checked by Claude Opus 5.5 on a crop and the whole frame, high-confidence boxes are checked by eye, and
   `select` writes the boxes to add.

`batch_run.py` runs passes 1 and 2 and the export on the Message Batches API at half price. Requests carry
their image inline and are split into batches of at most 200 MB; every stage is resumable and failed
requests are retried (3 attempts). `sheets.py` draws the review sheets used for the visual checks.

## Running it

From the repo root, with `ANTHROPIC_API_KEY=...` in `.env` (git-ignored):

```
python -m venv venv-vlm && venv-vlm/bin/pip install -r tools/vlm_annotation/requirements.txt
```

1. Write `configs/<name>.json` for the processed dataset `data/processing/<name>/` (fields:
   [configs/README.md](configs/README.md); examples: `obsea_2013_2014.json`, `kso_mussel_bruv.json`).
2. Pilot on about 25 random photos (a TSV with a `file_name` column; a few cents), look at the sheets, and
   estimate the cost of the whole dataset:
   ```
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run --images pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py static --run-dir data/vlm_annotation/<name>_pilot
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py test -n 5 --run-dir data/vlm_annotation/<name>_pilot
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py frames --images pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py estimate --run-dir data/vlm_annotation/<name>_pilot
   ```
   `estimate` extrapolates the pilot's measured cost per photo of each pass, and the Opus cost per box that
   `test` measured on 5 of its medium / low boxes, to the whole dataset, for each option: pass 1 only, pass 1
   + review, and either one + the Opus check of the medium / low boxes. On the two past runs it gives back their
   measured costs (KSO: passes $0.89, Opus check 993 boxes, $5.36 against $5.33 measured); for OBSEA the Opus
   check, never run, would have added about $273 for 50,582 boxes. Choose the option before going on.
3. Full run of the chosen option (in tmux; rerun the same command to resume):
   `VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run` (add `--stage pass1` for pass 1 only)
4. Cleaning, writing the visual decisions to a `decisions.json` (format in `verify_added.py`); `submit` and
   `collect` only with the Opus check (without it, only the high-confidence boxes are kept):
   ```
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py static
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py static
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py submit
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py collect
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py crops --confidence high
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py select --decisions decisions.json --out datasets/<name>_added_boxes.json
   VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py crops --added datasets/<name>_added_boxes.json --n 100
   ```

Outputs go to `data/vlm_annotation/<name>/`: `export/` (`<name>_vlm_coco.json`, `rejected_candidates.json`,
`review_queue.csv`, `summary.json`), `verify/` (static groups, Opus verdicts, cost), `sheets/`, and
`cost.json`. The dataset itself is not changed: its script reads `datasets/<name>_added_boxes.json`
(see the skill, step 7).

`pilot.py run/score/render` also compares several models or settings on hand-counted photos
(`--variants`); that is how tiled Haiku was chosen (see below).

## Past runs

| | OBSEA 2013-2014 (2026-10-20) | KSO Mussel Farm BRUV (2026-10-23) |
|---|---|---|
| photos | 14,952 (640x480, fixed camera on a reef) | 910 (640x360, GoPro over a mat with bait) |
| human boxes missing (hand count) | about 60% of the fish (40 photos) | 29% of the animals over 3 cm (64 frames) |
| passes 1 and 2 | $22.24 | $0.89 |
| result | 69,651 human boxes become 141,028 for human review (31% of the human boxes tightened) | 1,311 proposals cleaned to 572 added boxes (static objects, Opus check $5.33, eye checks) |
| check | 50 random photos: 39 better, 8 unchanged, 3 worse | 100 random added boxes: about 94 on an animal, none with the wrong class |

What these runs taught:

- **Tiles beat bigger models.** On 9 hand-counted photos, tiled Haiku found 82% of the human boxes and many
  missing fish at a fifteenth of Opus's cost; whole-photo boxes in schools are often offset.
- **The review removes real animals.** Most high and medium confidence pass-1 boxes the review removed were
  fish, so it may only remove low-confidence ones; low-confidence boxes are a coin flip either way.
- **Opus alone is too lenient on single frames**: in KSO it called algae fronds and fragments that sat still
  for 28 minutes "fish". The static-object check comes first, and every group is decided by eye.
- **Non-fish boxes are unreliable when the human boxes are all fish** (OBSEA: most of its 299 non-fish boxes
  were fish or white filaments).
- **Name the dataset's false positives in the setting** (KSO: snails and curled algae on the mat).

## Tests

The config-driven version of the scripts reproduces both past runs exactly, without calling the API: from
the stored batch results it rebuilds the same per-photo records (29,900 for OBSEA, 1,820 for KSO) and the
same export files as the hardcoded version on `gc/vlm-fish-annotation` (141,028 and 2,974 boxes, equal to
the committed exports), the same 48 static groups, and the same 572 added boxes as
`datasets/kso_mussel_bruv_added_boxes.json`. The pass-1, pass-2 and Opus requests it builds are identical
too (the Opus prompt was KSO's for every dataset before; it now comes from each config).

## Files

- `pilot.py`: pass 1, the config loading (`VLM_DATASET`), box matching and the model-comparison pilot
- `second_pass.py`: pass 2 and the merge
- `batch_run.py`: both passes on the Batch API, resumable; `status`, `cost`, and `estimate` (full-dataset cost per option, from a pilot)
- `export.py`: the COCO export for review
- `verify_added.py`: cleaning of the added boxes (static groups, Opus check, selection)
- `sheets.py`: review sheets (frames, crops, static groups)
- `configs/`: one config per dataset, and the field reference
- `requirements.txt`: `anthropic`, `numpy`, `opencv-python`, `pillow`
