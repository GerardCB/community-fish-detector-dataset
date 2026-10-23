---
name: pre-annotate
description: Add the boxes a processed dataset is missing with the Claude pre-annotation pipeline in tools/vlm_annotation (two Claude Haiku passes on the Message Batches API, then a cleaning step with an Opus check and visual checks). Use it when the user says a dataset's annotations are incomplete, asks to pre-annotate or "add missing boxes" with Claude or a VLM, or types /pre-annotate <dataset>.
---

# Pre-annotate a dataset with Claude

The user wants to add missing boxes to the processed dataset **$ARGUMENTS** (`data/processing/<name>/`).
The pipeline, its outputs and its two past runs (OBSEA, KSO Mussel Farm BRUV) are described in
`tools/vlm_annotation/README.md`; the config fields in `tools/vlm_annotation/configs/README.md`. Read both first.

Rules for the whole workflow:

- **Every step that calls the API costs money: state the estimate and wait for the user's go-ahead first.**
  Reference costs on the Batch API: both passes about $0.001-0.0015 per photo (OBSEA: $22 for 14,952
  photos; KSO: $0.89 for 910), the Opus check about $0.005 per box (KSO: $5.33 for 993 boxes).
- The API key is `ANTHROPIC_API_KEY` in `.env` (git-ignored). Never print it or commit it.
- Run the scripts from the repo root with `venv-vlm/bin/python` (create it with
  `python -m venv venv-vlm && venv-vlm/bin/pip install -r tools/vlm_annotation/requirements.txt`), with
  `VLM_DATASET=<name>`. Start long runs in a tmux session so the user can follow them.
- Check boxes by eye on the sheets of `sheets.py`, not by trusting a model: the Opus check alone is too
  lenient (in KSO it called algae fronds that sat still for 28 minutes "fish").
- Nothing is added to the dataset until the user has seen the precision check (step 6).

## 1. Measure the gap

Before spending anything, find out how many animals have no box. Pick about 60 photos at random, spread over
the dataset's videos, sites or deployments (6-8 per group), and look at each at 1.5x zoom with its human boxes
drawn. Count the animals clearly bigger than about 3 cm (fish and non-fish, as in CLAUDE.md) with and without a
box. Write the count to `tools/vlm_annotation/<name>_results/hand_count.tsv` (columns `file_name`, `group`,
`boxed`, `missed_over_3cm`, `note`) and estimate the missing boxes for the whole dataset, with a range.

Tell the user the miss rate and where the missing animals are (schools, busy frames, frame edges, background).
If only a few percent are missing, pre-annotation is probably not worth it: say so.

## 2. Write the config

Create `tools/vlm_annotation/configs/<name>.json` (see `configs/README.md`): `images_dir` and `annotations`
pointing at `data/processing/<name>/`, `prefix` "<name>_", `human_classes` (true if the dataset's boxes are
already fish / non-fish), the upscales for its photo size, and a `setting` that describes the scene, the usual
animals and their size, and what gets no box in this dataset. Fill `verify` too: the setting for the Opus
check, typical animals, the dataset's usual false boxes under `no_animal`, and `video_pattern` if the file
names hold a video or recording name. Show the config to the user.

## 3. Pilot (about $0.02-0.10)

List about 20 photos in a TSV with a `file_name` column (`tools/vlm_annotation/<name>_results/pilot.tsv`),
including some of the hand-counted ones and the hardest cases (schools, night, busy frames). After the go-ahead:

```
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run --images tools/vlm_annotation/<name>_results/pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py frames --images tools/vlm_annotation/<name>_results/pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
```

Look at every frame sheet: how many of the hand-counted missing animals were found, and what the false boxes
are. If the false boxes are a recurring object (algae, a cable, a snail), name it in `setting` and rerun the
pilot in a new run dir. Report found / missed / false boxes to the user.

## 4. Full run

`batch_run.py cost --run-dir data/vlm_annotation/<name>_pilot` gives the cost per photo; multiply by the number
of photos and ask. Then, in tmux:

```
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run
```

It is resumable (rerun the same command after an interruption) and ends with the export in
`data/vlm_annotation/<name>/export/` (COCO with `attributes.source`, `confidence` and `change` on every box).
Report the counts of `summary.json` and the measured cost.

## 5. Clean the added boxes

1. **Static objects**: `verify_added.py static`, then `sheets.py static`. For each group decide by eye whether
   it is an animal that sat still (crabs, flatfish) or an object (algae, debris, cable, equipment), and write
   the object indices to `object_groups` in `tools/vlm_annotation/<name>_results/decisions.json` (the animal
   ones to `animal_groups`). Add to `excluded_groups` any video whose added boxes should all go (for example
   one where only animals that get no box appear).
2. **Opus check of medium and low confidence boxes**: `verify_added.py test -n 3` shows three verdicts and
   their tokens; estimate the cost for all medium and low boxes and ask; then `verify_added.py submit` and
   `verify_added.py collect` (it waits for the batch).
3. **High-confidence boxes**: `sheets.py crops --confidence high` and look at every sheet. Record wrong
   classes in `class_fixes` (`{"a<annotation id>": "fish" | "non-fish"}`). If many high-confidence boxes are
   wrong in the same way, stop and tell the user: the setting probably needs work.
4. **Select**: `verify_added.py select --decisions tools/vlm_annotation/<name>_results/decisions.json
   --out datasets/<name>_added_boxes.json`. It drops boxes on a human box, static objects and excluded
   videos, keeps the high-confidence boxes and the medium / low ones Opus confirmed with high or medium
   confidence. Write a `note` in decisions.json saying how each choice was made.

## 6. Check the precision

`sheets.py crops --added datasets/<name>_added_boxes.json --n 100` draws 100 random boxes of the selection;
count by eye: on an animal with the right class, wrong class, not an animal, unclear. Compare the number of added boxes with the gap estimated in
step 1 to say how much of it is recovered.

## 7. Report and wire in

Tell the user the result before changing the dataset: boxes added (fish / non-fish), precision, share of the
gap recovered, total cost. If they agree, change `datasets/<name>.py` the way `kso_mussel_bruv.py` does
(branch `gc/kso_mussel_bruv`):

- load `datasets/<name>_added_boxes.json` (keyed by the original file name) in the conversion step
- add its boxes with the source categories `added fish` / `added non-fish`, mapped in `CATEGORIES_FILTER`
- give every annotation `attributes`: `{"source": "human"}` for the dataset's own boxes, and `source`
  (`vlm_pass1` or `vlm_review`), `confidence` and `verified` (the Opus verdict, if any) for the added ones,
  so they can be filtered later
- document in the script docstring, DATASETS.md, the README entry and the PR: the hand count, how the boxes
  were made and cleaned, the precision check, the remaining gap and the cost

Keep the small result files of the run in `tools/vlm_annotation/<name>_results/` (hand count, pilot list,
decisions, `cost.json`, `summary.json`, verdicts), and ask before committing anything.
