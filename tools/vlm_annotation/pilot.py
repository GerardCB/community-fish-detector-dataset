"""
Pass 1 of the pre-annotation, and a pilot to compare models on a few hand-counted photos.

The dataset (paths, scene description, classes of the human boxes) comes from the config named by
the VLM_DATASET environment variable (configs/<name>.json, or a path to a config file).

Pass 1 asks a Claude model to box every animal in a photo, independently of the existing (human)
boxes. A variant is a model, an effort level, and whether the photo is sent whole (upscaled
full_upscale times) or as 4 overlapping tiles (each upscaled tile_upscale times). With tiles, each
tile keeps the boxes whose center lies in its own quadrant of the photo, and duplicates are then
suppressed. batch_run.py uses the variant claude-haiku-5-5_tiled_v2.

The pilot compares the boxes of several variants with the existing ones, on photos listed in a TSV
(columns audit_id, file_name, clear_missed, possible_missed: a hand count of the unboxed animals):
  - existing boxes found again show whether the model finds and localizes the animals that
    annotators did box,
  - model boxes that match no existing box are candidate missing animals, to be reviewed,
  - token usage gives the cost per image and an estimate for the whole dataset.

Usage (from the repo root, with the API key in .env as ANTHROPIC_API_KEY=...):
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/pilot.py run --images list.tsv --variants claude-haiku-5-5_tiled_v2
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/pilot.py score --images list.tsv --variants ...
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/pilot.py render --images list.tsv --variants ...

Responses are cached in data/vlm_pilot/raw/<variant>/, so `run` only calls the API for
(variant, image) pairs that have no cached response yet.
"""

import argparse
import base64
import functools
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image

# USD per million tokens (input, output); thinking tokens are billed as output
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
VARIANTS = {
    "claude-haiku-5-5": {"model": "claude-haiku-5-5", "effort": "medium", "tiled": False},
    "claude-sonnet-5-5": {"model": "claude-sonnet-5-5", "effort": "medium", "tiled": False},
    "claude-opus-5-5": {"model": "claude-opus-5-5", "effort": "medium", "tiled": False},
    "claude-sonnet-5-5_high": {"model": "claude-sonnet-5-5", "effort": "high", "tiled": False},
    "claude-opus-5-5_high": {"model": "claude-opus-5-5", "effort": "high", "tiled": False},
    "claude-sonnet-5-5_tiled": {"model": "claude-sonnet-5-5", "effort": "medium", "tiled": True},
    "claude-opus-5-5_tiled": {"model": "claude-opus-5-5", "effort": "medium", "tiled": True},
    "claude-haiku-5-5_max": {"model": "claude-haiku-5-5", "effort": "max", "tiled": False},
    "claude-haiku-5-5_tiled": {"model": "claude-haiku-5-5", "effort": "medium", "tiled": True},
    "claude-haiku-5-5_tiled_high": {"model": "claude-haiku-5-5", "effort": "high", "tiled": True},
    # v2 prompt: fish and non-fish classes, following the dataset's annotation rules
    "claude-haiku-5-5_v2": {"model": "claude-haiku-5-5", "effort": "medium", "tiled": False, "prompt": "v2"},
    "claude-haiku-5-5_tiled_v2": {"model": "claude-haiku-5-5", "effort": "medium", "tiled": True, "prompt": "v2"},
    "claude-opus-5-5_v2": {"model": "claude-opus-5-5", "effort": "medium", "tiled": False, "prompt": "v2"},
    # cheaper Haiku tiled: less thinking, or tiles sent at 768x576 instead of 1152x864
    "claude-haiku-5-5_tiled_v2_low": {"model": "claude-haiku-5-5", "effort": "low", "tiled": True, "prompt": "v2"},
    "claude-haiku-5-5_tiled2x_v2": {"model": "claude-haiku-5-5", "effort": "medium", "tiled": True, "prompt": "v2",
                                     "tile_upscale": 2},
}
DEFAULT_VARIANTS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
TILE_FRACTION = 0.6  # each tile is 60% of the photo width and height (384x288 for 640x480), overlapping
NMS_IOU = 0.5
NMS_OVERLAP = 0.8
MATCH_IOU = 0.3
MATCH_OVERLAP = 0.6
MATCH_MAX_GROWTH = 4  # a model box up to 4x the area of an existing box it contains can be the same animal
MAX_TOKENS = 32000
CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}

# The dataset is chosen with the VLM_DATASET environment variable: the name of a config in configs/
# (configs/<name>.json) or the path of a config file. See configs/README.md for the fields.
CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_DEFAULTS = {"raw_images_dir": None, "prefix": "", "human_classes": False, "full_upscale": 2, "tile_upscale": 3,
                   "verify": {}}


def load_config() -> tuple[str, dict]:
    """(dataset name, config) from VLM_DATASET, with the defaults filled in."""
    chosen = os.environ.get("VLM_DATASET")
    if not chosen:
        known = ", ".join(sorted(c.stem for c in CONFIG_DIR.glob("*.json")))
        raise SystemExit(f"Set VLM_DATASET to a config name ({known}) or to the path of a config file")
    path = Path(chosen) if chosen.endswith(".json") else CONFIG_DIR / f"{chosen}.json"
    if not path.exists():
        raise SystemExit(f"No config {path}")
    config = {**CONFIG_DEFAULTS, **json.loads(path.read_text())}
    missing = {"images_dir", "annotations", "setting"} - set(config)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    return path.stem, config


DATASET, DATASET_CONFIG = load_config()
FULL_UPSCALE = DATASET_CONFIG["full_upscale"]  # whole photos are sent upscaled (2x suits 640x480 photos)
TILE_UPSCALE = DATASET_CONFIG["tile_upscale"]  # and tiles more (3x: 384x288 tiles are sent at 1152x864)
PILOT_DIR = Path("data/vlm_pilot")
IMAGES_DIR = Path(DATASET_CONFIG["images_dir"])
ANNOTATIONS_PATH = Path(DATASET_CONFIG["annotations"])
IMAGE_LIST = PILOT_DIR / "pilot_images.tsv"  # set with --images

SYSTEM_PROMPT = (
    "You annotate fish in underwater photos for the training set of a fish detector. "
    "Missing a fish is as harmful as boxing something that is not a fish."
)
SETTING = DATASET_CONFIG["setting"]
TASK = """Find every fish visible in the {subject}, including small, distant, blurry, partly \
hidden and night-time fish. In a school, box each fish separately.{edge}
Do not box anything that is not a fish: algae, drifting white filaments, particles, bubbles, \
the reef structure, ropes, crabs, octopus or jellyfish.

For each fish, give a tight bounding box in pixel coordinates of this {subject} (origin at the \
top-left corner; x to the right, y down) and a confidence: "high" if it is clearly a fish, \
"medium" if it is probably a fish, "low" if it is possibly a fish. Return an empty list if \
there is no fish."""
FULL_PROMPT = "This photo is {width}x{height} pixels. " + SETTING + "\n\n" + TASK.format(
    subject="photo", edge="", width="{width}", height="{height}"
)
TILE_PROMPT = (
    "This image ({width}x{height} pixels) is an enlarged crop of a photo. The photo "
    + SETTING.replace("It comes", "comes")
    + "\n\n"
    + TASK.format(subject="image", edge=" For a fish cut by the edge of the image, box its visible part.")
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "fish": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["x_min", "y_min", "x_max", "y_max", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fish"],
    "additionalProperties": False,
}

# v2: the dataset's annotation rules (every free-moving animal over ~3 cm, fish or non-fish)
SYSTEM_PROMPT_V2 = (
    "You annotate animals in underwater photos for the training set of a fish detector. "
    "Missing an animal is as harmful as boxing something that should not have a box."
)
TASK_V2 = """Box every animal in the {subject} that moves on its own and is bigger than about \
3 cm, with its class:
- "fish": any cartilaginous, ray-finned or bony fish (sharks, rays, eels, groupers, sea breams...).
- "non-fish": any other free-moving marine animal clearly bigger than about 3 cm: turtles, \
dolphins, seals, crabs, lobsters, shrimp, octopus, squid, cuttlefish, jellyfish, starfish, sea \
urchins, sea cucumbers.
Box an animal even if only part of it is visible (a head, a tail, a fin){edge}, and also small, \
distant, blurry and night-time animals. In a school, box each animal separately. Each box must \
fit its animal tightly.
Do not box anything that does not move on its own (corals, sponges, anemones, barnacles, shells, \
algae, rocks, debris, ropes, the reef structure, divers, equipment), drifting white filaments, \
particles or bubbles, animals smaller than about 3 cm (fry, small shrimp, plankton), or anything \
you are not sure moves on its own.

Give each box in pixel coordinates of this {subject} (origin at the top-left corner; x to the \
right, y down), with its class and a confidence: "high" if you are sure it is an animal of that \
class, "medium" if it probably is, "low" if it possibly is. Return an empty list if there is no \
animal."""
FULL_PROMPT_V2 = "This photo is {width}x{height} pixels. " + SETTING + "\n\n" + TASK_V2.format(
    subject="photo", edge=""
)
TILE_PROMPT_V2 = (
    "This image ({width}x{height} pixels) is an enlarged crop of a photo. The photo "
    + SETTING.replace("It comes", "comes")
    + "\n\n"
    + TASK_V2.format(subject="image", edge=", or it is cut by the edge of the image")
)
OUTPUT_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "animals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string", "enum": ["fish", "non-fish"]},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["class", "x_min", "y_min", "x_max", "y_max", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["animals"],
    "additionalProperties": False,
}
PROMPTS = {
    "v1": {"system": SYSTEM_PROMPT, "full": FULL_PROMPT, "tile": TILE_PROMPT, "schema": OUTPUT_SCHEMA, "key": "fish"},
    "v2": {"system": SYSTEM_PROMPT_V2, "full": FULL_PROMPT_V2, "tile": TILE_PROMPT_V2, "schema": OUTPUT_SCHEMA_V2,
           "key": "animals"},
}


def load_env_file(path: Path = Path(".env")):
    """Set variables from a KEY=VALUE .env file, without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pilot_images() -> list[dict]:
    lines = Path(IMAGE_LIST).read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def raw_path(variant: str, file_name: str) -> Path:
    return PILOT_DIR / "raw" / variant / f"{Path(file_name).stem}.json"


def encode_image(image: Image.Image, crop: tuple[int, int, int, int], upscale: int) -> tuple[str, int, int]:
    """Crop of the photo, upscaled, as base64 JPEG, and its size."""
    region = image.crop(crop)
    region = region.resize((region.width * upscale, region.height * upscale), Image.LANCZOS)
    buffer = io.BytesIO()
    region.save(buffer, format="JPEG", quality=92)
    return base64.standard_b64encode(buffer.getvalue()).decode(), region.width, region.height


def tiles(width: int, height: int) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float]]]:
    """2x2 overlapping tiles: (crop box, quadrant whose box centers the tile keeps)."""
    tile_w, tile_h = round(width * TILE_FRACTION), round(height * TILE_FRACTION)
    mid_x, mid_y = width / 2, height / 2
    result = []
    for left, (q_x0, q_x1) in ((0, (0, mid_x)), (width - tile_w, (mid_x, width))):
        for top, (q_y0, q_y1) in ((0, (0, mid_y)), (height - tile_h, (mid_y, height))):
            result.append(((left, top, left + tile_w, top + tile_h), (q_x0, q_y0, q_x1, q_y1)))
    return result


def request_params(model: str, effort: str, prompts: dict, data: str, prompt: str) -> dict:
    """Messages API parameters for one image (base64 JPEG) and its prompt, with a JSON-schema answer."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": prompts["system"],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": prompts["schema"]}},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def parse_message(message, key: str | None) -> tuple[dict, object]:
    """Call record and parsed answer (its `key` field, or all of it) of a response; None on failure."""
    call = {
        "model_served": message.model,
        "stop_reason": message.stop_reason,
        "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
    }
    if message.stop_reason != "end_turn":
        call["error"] = f"stop_reason {message.stop_reason}"
        return call, None
    text = next((block.text for block in message.content if block.type == "text"), "")
    call["text"] = text
    try:
        parsed = json.loads(text)
        return call, parsed[key] if key else parsed
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        call["error"] = f"unparseable output: {e}"
        return call, None


def ask(client: anthropic.Anthropic, params: dict, key: str | None) -> tuple[dict, object]:
    """One streamed request. Returns (call record, parsed answer or None on failure)."""
    started = time.time()
    try:
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()
    except anthropic.BadRequestError as e:
        return {"error": f"bad request: {e.message}"}, None
    except anthropic.RateLimitError as e:
        return {"error": f"rate limited: {e.message}"}, None
    except anthropic.APIStatusError as e:
        return {"error": f"API error {e.status_code}: {e.message}"}, None
    except anthropic.APIConnectionError as e:
        return {"error": f"connection error: {e}"}, None
    call, parsed = parse_message(message, key)
    call["elapsed_s"] = round(time.time() - started, 1)
    return call, parsed


def to_photo_box(fish: dict, crop, upscale, width, height):
    """Box of the model (in sent-image pixels) in photo pixels, clipped; None if degenerate."""
    x0, x1 = sorted((fish["x_min"] / upscale + crop[0], fish["x_max"] / upscale + crop[0]))
    y0, y1 = sorted((fish["y_min"] / upscale + crop[1], fish["y_max"] / upscale + crop[1]))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    return [x0, y0, x1, y1] if x1 - x0 > 1 and y1 - y0 > 1 else None


def suppress_duplicates(boxes: list[dict]) -> list[dict]:
    """Drop boxes that duplicate a kept box, keeping higher confidence first, then larger."""
    order = sorted(
        boxes,
        key=lambda b: (CONFIDENCE_RANK[b["confidence"]], (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])),
        reverse=True,
    )
    kept = []
    for b in order:
        if all(iou(b["box"], k["box"]) < NMS_IOU and overlap(b["box"], k["box"]) < NMS_OVERLAP for k in kept):
            kept.append(b)
    return kept


def pass1_jobs(variant: str, width: int, height: int) -> list[dict]:
    """The requests of one photo: the whole photo, or 4 overlapping tiles."""
    config = VARIANTS[variant]
    prompts = PROMPTS[config.get("prompt", "v1")]
    if config["tiled"]:
        upscale = config.get("tile_upscale", TILE_UPSCALE)
        return [{"crop": crop, "quadrant": quadrant, "upscale": upscale, "prompt": prompts["tile"]}
                for crop, quadrant in tiles(width, height)]
    return [{"crop": (0, 0, width, height), "quadrant": None, "upscale": FULL_UPSCALE, "prompt": prompts["full"]}]


def pass1_request(variant: str, photo: Image.Image, job: dict) -> dict:
    config = VARIANTS[variant]
    prompts = PROMPTS[config.get("prompt", "v1")]
    data, sent_w, sent_h = encode_image(photo, job["crop"], job["upscale"])
    return request_params(config["model"], config["effort"], prompts, data, job["prompt"].format(width=sent_w, height=sent_h))


def pass1_key(variant: str) -> str:
    return PROMPTS[VARIANTS[variant].get("prompt", "v1")]["key"]


def pass1_merge(variant: str, file_name: str, width: int, height: int, jobs: list[dict], outcomes: list) -> dict:
    """
    Record of one photo from the (call, answer) of its requests, in job order: boxes in photo
    pixels (each tile keeps the boxes centered in its quadrant; duplicates are suppressed).
    """
    config = VARIANTS[variant]
    record = {"variant": variant, **config, "file_name": file_name, "calls": []}
    boxes = []
    for job, (call, fish) in zip(jobs, outcomes):
        call["crop"] = job["crop"]
        record["calls"].append(call)
        if fish is None:
            record["error"] = call["error"]
            break
        crop, quadrant, upscale = job["crop"], job["quadrant"], job["upscale"]
        for f in fish:
            box = to_photo_box(f, crop, upscale, width, height)
            if box is None:
                continue
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if quadrant and not (quadrant[0] <= center_x < quadrant[2] and quadrant[1] <= center_y < quadrant[3]):
                continue
            boxes.append({"box": box, "confidence": f["confidence"], "class": f.get("class", "fish")})
    if len(record["calls"]) < len(jobs) and "error" not in record:
        record["error"] = "missing results"
    record["usage"] = {
        key: sum(c.get("usage", {}).get(key, 0) for c in record["calls"]) for key in ("input_tokens", "output_tokens")
    }
    record["elapsed_s"] = round(sum(c.get("elapsed_s", 0) for c in record["calls"]), 1)
    if "error" not in record:
        record["boxes"] = suppress_duplicates(boxes) if config["tiled"] else boxes
    return record


def annotate(client: anthropic.Anthropic, variant: str, file_name: str) -> dict:
    """Fish boxes of one photo for one variant, in photo pixels, with the summed token usage."""
    with Image.open(IMAGES_DIR / file_name) as photo:
        photo = photo.convert("RGB")
        width, height = photo.size
        jobs = pass1_jobs(variant, width, height)
        outcomes = []
        for job in jobs:
            call, fish = ask(client, pass1_request(variant, photo, job), pass1_key(variant))
            outcomes.append((call, fish))
            if fish is None:
                break
    return pass1_merge(variant, file_name, width, height, jobs, outcomes)


def run(args):
    load_env_file()
    client = anthropic.Anthropic()
    images = pilot_images()
    jobs = [
        (variant, image["file_name"])
        for variant in args.variants
        for image in images
        if not raw_path(variant, image["file_name"]).exists()
    ]
    print(f"{len(jobs)} photos to annotate ({len(args.variants) * len(images) - len(jobs)} cached)")

    def job(variant_and_file):
        variant, file_name = variant_and_file
        record = annotate(client, variant, file_name)
        if "error" in record and not record["usage"]["input_tokens"]:
            print(f"  {variant} {file_name[16:31]}: {record['error']} (not cached)")
            return
        path = raw_path(variant, file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        print(f"  {variant} {file_name[16:31]}: {len(record.get('boxes', []))} fish, {record['elapsed_s']}s"
              + (f", {record['error']}" if "error" in record else ""))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, jobs))


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap(a, b) -> float:
    """Intersection over the smaller box's area: 1 when one box lies inside the other."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def match(model_boxes, existing_boxes):
    """
    Greedy one-to-one matching. A model box matches an existing box when their IoU is at least
    MATCH_IOU, or when one lies mostly inside the other (overlap >= MATCH_OVERLAP): many existing
    boxes are much looser than the models' boxes. Returns {model index: (existing index, IoU)}.
    """
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    pairs = []
    for i, m in enumerate(model_boxes):
        for j, e in enumerate(existing_boxes):
            value_iou = iou(m, e)
            # A model box much bigger than an existing box (e.g. a big fish with a small fish's
            # box inside it) is a different animal; a loose existing box around a tight model box
            # is the same one
            nested = overlap(m, e) >= MATCH_OVERLAP and area(m) <= MATCH_MAX_GROWTH * area(e)
            if value_iou >= MATCH_IOU or nested:
                pairs.append((value_iou + overlap(m, e), value_iou, i, j))
    used_m, used_e, matches = set(), set(), {}
    for _, value_iou, i, j in sorted(pairs, reverse=True):
        if i not in used_m and j not in used_e:
            used_m.add(i)
            used_e.add(j)
            matches[i] = (j, value_iou)
    return matches


@functools.lru_cache(maxsize=1)
def existing_boxes_by_file() -> dict[str, list[list[float]]]:
    coco = json.loads(ANNOTATIONS_PATH.read_text())
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    boxes = {}
    for a in human_annotations(coco):
        x, y, w, h = a["bbox"]
        boxes.setdefault(names[a["image_id"]], []).append([x, y, x + w, y + h])
    return boxes


def human_annotations(coco: dict) -> list[dict]:
    """The boxes drawn by people: boxes added by an earlier run carry attributes.source vlm_*."""
    return [a for a in coco["annotations"] if a.get("attributes", {}).get("source", "human") == "human"]


def dataset_image_count() -> int:
    """Photos in the dataset, to extrapolate costs."""
    return len(json.loads(ANNOTATIONS_PATH.read_text())["images"])


def existing_classes_by_file() -> dict[str, list[str]]:
    """The class ("fish" / "non-fish") of each existing box, in the order of existing_boxes_by_file."""
    if not DATASET_CONFIG["human_classes"]:
        # all fish; existing_boxes_by_file may be the frozen copy of a batch run
        return {name: ["fish"] * len(boxes) for name, boxes in existing_boxes_by_file().items()}
    coco = json.loads(ANNOTATIONS_PATH.read_text())
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    classes = {}
    for a in human_annotations(coco):
        classes.setdefault(names[a["image_id"]], []).append(categories[a["category_id"]])
    return classes


def score(args):
    existing = existing_boxes_by_file()
    images = pilot_images()
    rows = []
    for variant in args.variants:
        found = total_existing = new = 0
        new_by_conf = {"high": 0, "medium": 0, "low": 0}
        ious, cost, seconds, errors, done = [], 0.0, 0.0, 0, 0
        per_image, non_fish = [], []
        price_in, price_out = PRICES[VARIANTS[variant]["model"]]
        for image in images:
            path = raw_path(variant, image["file_name"])
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            usage = record.get("usage", {})
            cost += (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6
            seconds += record.get("elapsed_s", 0)
            if "error" in record:
                errors += 1
                continue
            done += 1
            # The existing annotations are fish only, so only fish boxes are matched against them
            fish_boxes = [b for b in record["boxes"] if b.get("class", "fish") == "fish"]
            non_fish += [b for b in record["boxes"] if b.get("class", "fish") == "non-fish"]
            gt = existing.get(image["file_name"], [])
            matches = match([b["box"] for b in fish_boxes], gt)
            found += len(matches)
            total_existing += len(gt)
            ious += [v for _, v in matches.values()]
            image_new = 0
            for i, b in enumerate(fish_boxes):
                if i not in matches:
                    new_by_conf[b["confidence"]] += 1
                    image_new += 1
            new += image_new
            per_image.append((image["audit_id"], len(gt), len(matches), image_new, int(image["clear_missed"])))
        if not done and not errors:
            continue
        n_photos = done + errors
        rows.append(
            {
                "variant": variant,
                "images": done,
                "errors": errors,
                "existing_found": f"{found}/{total_existing} ({found / max(1, total_existing):.0%})",
                "mean_iou_of_found": round(float(np.mean(ious)), 2) if ious else None,
                "new_boxes": new,
                "new_high/medium/low": f"{new_by_conf['high']}/{new_by_conf['medium']}/{new_by_conf['low']}",
                "non_fish_high/medium/low": "/".join(
                    str(sum(b["confidence"] == c for b in non_fish)) for c in ("high", "medium", "low")
                ),
                "pilot_cost_usd": round(cost, 3),
                "avg_s_per_image": round(seconds / n_photos, 1),
                "est_full_dataset_usd": round(cost / n_photos * dataset_image_count()),
                "est_full_dataset_batch_usd": round(cost / n_photos * dataset_image_count() / 2),
                "per_image": per_image,
            }
        )
    clear_missed = sum(int(i["clear_missed"]) for i in images)
    possible = sum(int(i["possible_missed"]) for i in images)
    if all(int(i["clear_missed"]) >= 0 for i in images):
        print(f"Hand count on these {len(images)} images: {sum(len(existing.get(i['file_name'], [])) for i in images)} "
              f"existing boxes, {clear_missed} clearly unboxed fish (+{possible} possible)\n")
    for row in rows:
        print(f"{row['variant']}: {row['images']} images, {row['errors']} errors")
        for key in ["existing_found", "mean_iou_of_found", "new_boxes", "new_high/medium/low",
                    "non_fish_high/medium/low", "pilot_cost_usd", "avg_s_per_image", "est_full_dataset_usd",
                    "est_full_dataset_batch_usd"]:
            print(f"  {key}: {row[key]}")
        print("  per image (audit id: existing found / existing, new boxes vs hand-counted missing):")
        print("   " + ", ".join(f"#{a}: {f}/{g}, {n} vs {c}" for a, g, f, n, c in row["per_image"]))
    (PILOT_DIR / "scores.json").write_text(json.dumps(rows, indent=1))


def render(args):
    """One sheet per photo: existing boxes, then each variant's boxes (matched cyan, new by confidence)."""
    existing = existing_boxes_by_file()
    out_dir = PILOT_DIR / args.sheets_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"high": (255, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 255)}
    for image in pilot_images():
        source = cv2.imread(str(IMAGES_DIR / image["file_name"]))
        gt = existing.get(image["file_name"], [])
        scale = 1.5
        panels = []
        base = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        panel = base.copy()
        for b in gt:
            cv2.rectangle(panel, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)), (0, 220, 0), 2)
        label = f"existing: {len(gt)} boxes (hand count: +{image['clear_missed']} clearly missing)"
        panels.append((panel, label))

        for variant in args.variants:
            panel = base.copy()
            path = raw_path(variant, image["file_name"])
            record = json.loads(path.read_text()) if path.exists() else {"error": "not run"}
            if "error" in record:
                label = f"{variant}: {record['error']}"
            else:
                boxes = [b for b in record["boxes"] if b.get("class", "fish") == "fish"]
                others = [b for b in record["boxes"] if b.get("class", "fish") == "non-fish"]
                matches = match([b["box"] for b in boxes], gt)
                for i, b in enumerate(boxes):
                    x0, y0, x1, y1 = (int(v * scale) for v in b["box"])
                    color = (255, 255, 0) if i in matches else colors[b["confidence"]]
                    cv2.rectangle(panel, (x0, y0), (x1, y1), color, 2)
                for b in others:
                    x0, y0, x1, y1 = (int(v * scale) for v in b["box"])
                    cv2.rectangle(panel, (x0, y0), (x1, y1), (0, 0, 255), 3)
                label = f"{variant}: {len(boxes)} fish, {len(matches)} on existing, {len(boxes) - len(matches)} new"
                if others:
                    label += f"; {len(others)} non-fish (red)"
            panels.append((panel, label))

        tiles_ = []
        for panel, label in panels:
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles_.append(panel)
        while len(tiles_) % 2:
            tiles_.append(np.zeros_like(tiles_[0]))
        sheet = np.vstack([np.hstack(tiles_[i:i + 2]) for i in range(0, len(tiles_), 2)])
        legend = np.full((34, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(legend, f"#{image['audit_id']}  green = existing; model boxes: cyan = matches an existing box, "
                    "new: magenta = high, orange = medium, yellow = low confidence",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"pilot_{int(image['audit_id']):02d}.jpg"), np.vstack([legend, sheet]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Sheets written to {out_dir}")


def review(args):
    """Sheet per variant of random NEW boxes (no existing box there), 10 per confidence, to judge by eye."""
    import random

    existing = existing_boxes_by_file()
    out_dir = PILOT_DIR / args.sheets_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"high": (255, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 255)}
    size = 220
    for variant in args.variants:
        pool = {"high": [], "medium": [], "low": []}
        for image in pilot_images():
            record = json.loads(raw_path(variant, image["file_name"]).read_text())
            fish_boxes = [b for b in record["boxes"] if b.get("class", "fish") == "fish"]
            matches = match([b["box"] for b in fish_boxes], existing.get(image["file_name"], []))
            new_boxes = [b for i, b in enumerate(fish_boxes) if i not in matches]
            new_boxes += [b for b in record["boxes"] if b.get("class", "fish") == "non-fish"]
            for b in new_boxes:
                pool[b["confidence"]].append((image["file_name"], b["box"], b.get("class", "fish")))
        rng = random.Random(5)
        picks = [(conf, *p) for conf in pool for p in rng.sample(pool[conf], min(10, len(pool[conf])))]
        tiles_ = []
        for k, (conf, file_name, (x0, y0, x1, y1), cls) in enumerate(picks, 1):
            photo = cv2.imread(str(IMAGES_DIR / file_name))
            height, width = photo.shape[:2]
            pad = max(20, 0.8 * max(x1 - x0, y1 - y0))
            cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            cx1, cy1 = int(min(width, x1 + pad)), int(min(height, y1 + pad))
            crop = photo[cy0:cy1, cx0:cx1]
            s = size / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (int((x0 - cx0) * s), int((y0 - cy0) * s)), (int((x1 - cx0) * s), int((y1 - cy0) * s)),
                          colors[conf], 2)
            tile = np.full((size + 22, size, 3), 30, np.uint8)
            tile[:crop.shape[0], :crop.shape[1]] = crop
            label = f"#{k} {conf}" + (" NON-FISH" if cls == "non-fish" else "")
            cv2.putText(tile, label, (4, size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles_.append(tile)
        while len(tiles_) % 6:
            tiles_.append(np.full_like(tiles_[0], 30))
        sheet = np.vstack([np.hstack(tiles_[i:i + 6]) for i in range(0, len(tiles_), 6)])
        head = np.full((32, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{variant}: random NEW boxes (no existing box there), 10 per confidence",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"new_boxes_{variant}.jpg"), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Review sheets written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "score", "render", "review"])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sheets-dir", default="sheets")
    parser.add_argument("--images", default=str(IMAGE_LIST), help="TSV list of photos (default: the 9 pilot photos)")
    args = parser.parse_args()
    globals()["IMAGE_LIST"] = Path(args.images)
    {"run": run, "score": score, "render": render, "review": review}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
