# Dataset configs

One JSON file per dataset, chosen with `VLM_DATASET=<name>` (this folder) or `VLM_DATASET=path/to/config.json`.
The scripts read the human boxes from `annotations` and never change the dataset itself.

| field | required | meaning |
|---|---|---|
| `images_dir` | yes | folder with the photos, named as in `annotations` (usually `data/processing/<name>/JPEGImages`) |
| `annotations` | yes | COCO file with the human boxes (usually `data/processing/<name>/annotations_coco.json`). Boxes whose `attributes.source` is not `human` (added by an earlier run) are ignored |
| `setting` | yes | what the photos show, inserted in the pass-1 and pass-2 prompts after "This photo is WxH pixels." Starts with "It comes from ..." (see below) |
| `human_classes` | no (false) | true if the human boxes have the categories `fish` / `non-fish`; false if they are all fish |
| `prefix` | no ("") | file-name prefix added by the dataset script; stripped to find photos in `raw_images_dir` and in the keys of the added-boxes file |
| `raw_images_dir` | no | the same photos under their original names (without `prefix`), read instead of `images_dir` when present |
| `full_upscale` | no (2) | factor by which whole photos are enlarged when sent (pass 2 on uncrowded photos) |
| `tile_upscale` | no (3) | factor by which the 4 tiles (each 60% of the photo's width and height) are enlarged (pass 1, and pass 2 on photos with more than 20 boxes) |
| `verify` | no | the Opus check of added boxes (`verify_added.py`), see below |
| `description` | no | a note for people |

`verify` fields, all optional:

| field | default | meaning |
|---|---|---|
| `setting` | `setting`, with "It comes" changed to "The photos come" | first paragraph of the Opus prompt |
| `fish_examples` | "of any kind" | examples in the "fish" label |
| `non_fish_examples` | crabs, shrimp, starfish... | examples in the "non-fish" label |
| `no_animal` | algae, debris, ropes, shadows... | what counts as "no animal" in this dataset: name its typical false boxes |
| `video_pattern` | none (one video) | regex whose first group is the video (or recording, camera) of a file name, for the static-object check, e.g. `^(.*)_frame_\\d+\\.jpg$`. Leave it out for a single fixed camera |

## Writing the setting

The model sees each photo with no other context, so the setting should say:

- the camera and the scene: fixed camera, BRUV, ROV, diver; reef, sand, mat with bait; depth, sea, day or night
- the animals that usually appear, and their size in the photo when they are small ("a goby is only a few dozen pixels long")
- what gets no box in this dataset because it looks like an animal: sessile or small animals (snails, anemones), algae fronds, debris, filaments, equipment

Keep it to a few sentences, in the style of `kso_mussel_bruv.json`. Test it on a pilot of about 20 photos before a full run (see the `/pre-annotate` skill).

## Choosing the upscales

Small animals need enough pixels, but larger images cost more tokens and the API scales very large ones down.
Aim for what the two runs sent: whole photos of about 1,280 px and tiles of about 1,150 px on the long side:

| photo size | `full_upscale` | `tile_upscale` |
|---|---|---|
| 640x480 or 640x360 | 2 | 3 |
| 1280x720 | 1 | 2 (tiles 768x432, sent at 1536x864) |
| 1920x1080 | 1 | 1 (tiles 1152x648) |
