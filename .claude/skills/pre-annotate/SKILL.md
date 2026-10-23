---
name: pre-annotate
description: Add the boxes a processed dataset is missing with the Claude pre-annotation pipeline in tools/vlm_annotation (two Claude Haiku passes on the Message Batches API, then a cleaning step with an Opus check and visual checks). Use it when the user says a dataset's annotations are incomplete, asks to pre-annotate or "add missing boxes" with Claude or a VLM, or types /pre-annotate <dataset>.
---

# Pre-annotate a dataset with Claude

The user wants to add missing boxes to the processed dataset **$ARGUMENTS** (`data/processing/<name>/`).
The pipeline, its outputs and its two past runs (OBSEA, KSO Mussel Farm BRUV) are described in
`tools/vlm_annotation/README.md`; the config fields in `tools/vlm_annotation/configs/README.md`. Read both first.

Rules for the whole workflow:

- **Always pilot first, and never run anything at full scale without the user's explicit go-ahead.** The pilot
  (step 3) costs cents: say how much and run it. It measures the cost of every stage, `batch_run.py estimate`
  turns that into the cost of each way of running the whole dataset, and the workflow stops there until the
  user has chosen an option and said yes. Reference costs on the Batch API: both passes about
  $0.001-0.0015 per photo (OBSEA: $22 for 14,952 photos; KSO: $0.89 for 910), the Opus check about $0.005 per
  checked box, which can dominate (KSO: $5.33 for 993 boxes; on OBSEA it would have been about $270).
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

## 3. Pilot and cost estimate (a few cents)

Use about 25 of the hand-counted photos of step 1 as the pilot: they are a random sample (so the cost
extrapolates) and you know how many animals in them have no box (so you can measure what the pipeline finds).
Write them to `tools/vlm_annotation/<name>_results/pilot.tsv` (a `file_name` column). Tell the user the pilot
costs a few cents (about $0.0015 per photo for both passes on the Batch API, plus about $0.01 per Opus test
request, sent directly) and run it; small batches usually finish within minutes:

```
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run --images tools/vlm_annotation/<name>_results/pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py static --run-dir data/vlm_annotation/<name>_pilot
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py test -n 5 --run-dir data/vlm_annotation/<name>_pilot
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py frames --images tools/vlm_annotation/<name>_results/pilot.tsv --run-dir data/vlm_annotation/<name>_pilot
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py estimate --run-dir data/vlm_annotation/<name>_pilot
```

The first command runs pass 1 and the review on the pilot photos; `test` sends 5 of their medium / low boxes to
the Opus check and records its cost per box; `estimate` extrapolates the measured costs to the whole dataset.

Look at every frame sheet and count, against the hand count: unboxed animals found, still missed, and false
boxes (and what they are). If the false boxes are a recurring object (algae, a cable, a snail), name it in
`setting` and run the pilot again in a new run dir.

**Then stop and ask.** Show the user:

- what the pilot found: unboxed animals found / missed, false boxes and what they were, its measured cost
- the estimate table of `batch_run.py estimate`, one row per option, on the Batch API (and the standard API):
  - pass 1 only: proposals without the review
  - pass 1 + review (pass 2): the review tightens loose human boxes, removes doubtful proposals and adds missed
    animals
  - either one + the Opus check of the medium / low confidence boxes; without it, only the high-confidence boxes
    (checked by eye) are added
- your recommendation, with the reason (for example: the review is worth its cost when the human boxes are
  loose; the Opus check when there are many medium / low boxes and their pilot precision is low)

Ask which option to run and whether to go ahead. Do not start step 4 until the user answers yes to an option.

## 4. Full run (only the option the user chose)

In tmux, for all photos:

```
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run                # pass 1 + review
VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run --stage pass1  # pass 1 only
```

It is resumable (rerun the same command after an interruption) and ends with the export in
`data/vlm_annotation/<name>/export/` (COCO with `attributes.source`, `confidence` and `change` on every box).
Report the counts of `summary.json` and the measured cost next to the estimate.

## 5. Clean the added boxes

1. **Static objects** (no API): `verify_added.py static`, then `sheets.py static`. For each group decide by eye
   whether it is an animal that sat still (crabs, flatfish) or an object (algae, debris, cable, equipment), and
   write the object indices to `object_groups` in `tools/vlm_annotation/<name>_results/decisions.json` (the
   animal ones to `animal_groups`). Add to `excluded_groups` any video whose added boxes should all go (for
   example one where only animals that get no box appear).
2. **Opus check**, only if the chosen option includes it: `verify_added.py submit`, then `verify_added.py
   collect` (it waits for the batch). If the number of medium / low boxes in the full run is well above the
   estimate (more than about 20%), tell the user the new cost and ask again before submitting.
3. **High-confidence boxes**: `sheets.py crops --confidence high` and look at every sheet. Record wrong
   classes in `class_fixes` (`{"a<annotation id>": "fish" | "non-fish"}`). If many high-confidence boxes are
   wrong in the same way, stop and tell the user: the setting probably needs work.
4. **Select**: `verify_added.py select --decisions tools/vlm_annotation/<name>_results/decisions.json
   --out datasets/<name>_added_boxes.json`. It drops boxes on a human box, static objects and excluded
   videos, keeps the high-confidence boxes and, with the Opus check, the medium / low ones Opus confirmed with
   high or medium confidence. Write a `note` in decisions.json saying how each choice was made.

## 6. Check the precision

`sheets.py crops --added datasets/<name>_added_boxes.json --n 100` draws 100 random boxes of the selection;
count by eye: on an animal with the right class, wrong class, not an animal, unclear. Compare the number of
added boxes with the gap estimated in step 1 to say how much of it is recovered.

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
