"""
Second pass: a model reviews the boxes of a photo against the annotation checklist.

The model sees the photo with every current box drawn and numbered: the dataset's existing boxes
plus the boxes a first-pass variant added (pilot.py). For each box it returns a verdict (keep,
remove, or fix with new coordinates and/or class), then the animals that still have no box, and
whether the photo is unusable.

Reviewer versions:
  v1: one request on the whole photo (2x); all boxes look the same.
  v2: the dataset's boxes (annotated by people) are drawn in green and the first pass's in
      yellow, and the prompt says to remove a person's box only when it is clearly wrong; the
      code also ignores the removal of a person's box unless the model is highly confident.
      Photos with more than TILE_THRESHOLD boxes are reviewed in the 4 overlapping tiles of
      pilot.py (3x): a box is judged by the tile whose quadrant holds its center, and added
      animals are kept only in their tile's quadrant and when they duplicate no kept box.

Usage (from the repo root):
  venv-vlm/bin/python tools/vlm_annotation/second_pass.py run --reviewers review_haiku_v2
  venv-vlm/bin/python tools/vlm_annotation/second_pass.py render --reviewers review_haiku_v2
"""

import argparse
import base64
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import pilot  # noqa: E402

FIRST_PASS = "claude-haiku-5-5_tiled_v2"
REVIEWERS = {
    "review_haiku": {"model": "claude-haiku-5-5", "effort": "medium", "version": 1},
    "review_haiku_v2": {"model": "claude-haiku-5-5", "effort": "medium", "version": 2},
}
UPSCALE = 2
TILE_THRESHOLD = 20  # v2: photos with more boxes are reviewed in tiles

SYSTEM_PROMPT = "You review the bounding-box annotations of underwater photos for the training set of a fish detector."
CLASSES = """- "fish": any cartilaginous, ray-finned or bony fish (sharks, rays, eels, groupers, sea breams...).
- "non-fish": any other free-moving marine animal clearly bigger than about 3 cm: turtles, \
dolphins, whales, seals, crabs, lobsters, shrimp, octopus, squid, cuttlefish, jellyfish, starfish, \
sea urchins, sea cucumbers.
- No box: anything that does not move on its own (corals, sponges, anemones, barnacles, shells, \
algae, rocks, debris, ropes, the reef structure, divers, equipment), drifting white filaments, \
particles, bubbles, shadows, sand, animals smaller than about 3 cm (fry, small shrimp, plankton), \
and anything you are not sure moves on its own."""
PROMPT = """This photo is {width}x{height} pixels. """ + pilot.SETTING + """ It shows {n} numbered \
yellow boxes, listed below with their class and pixel coordinates in this photo \
(x_min, y_min, x_max, y_max; origin at the top-left corner).

A photo is correctly annotated when every fish and every free-moving marine animal larger than \
about 3 cm has one box of the right class, even if only part of it is visible (a tail, a fin), \
each box fits its animal tightly (small gaps are fine), and nothing else has a box.
""" + CLASSES + """

Review every box and give a verdict:
- "remove": an extra box, on something that should not have one, or a second box on an animal \
that already has one.
- "fix": a box that is clearly too big, too small, cut off or shifted (not for small gaps), a box \
that covers two animals (fix it to cover one of them and add the other under "missing"), or a \
box with the wrong class. Give the corrected coordinates and class.
- "keep": anything else; repeat its coordinates and class unchanged.
Then list under "missing" every animal that has no box: check the edges, the background and \
behind the reef. Give its class, tight coordinates and a confidence ("high", "medium" or "low").
If the photo is too blurry or dark to judge, is corrupted or is not underwater, set "unusable" \
to true; then nothing else needs marking.

Boxes:
{boxes}"""

PROMPT_V2 = """This {subject} is {width}x{height} pixels.{crop_note} """ + pilot.SETTING + """ It \
shows {n} numbered boxes, listed below with their class, pixel coordinates in this {subject} \
(x_min, y_min, x_max, y_max; origin at the top-left corner) and who made them:
- green boxes were annotated by people: keep them unless they are clearly wrong. A faint, blurry, \
dark or partly visible animal is not a reason to remove its box; fix a box that is clearly too \
loose instead of removing it.
- yellow boxes were proposed by a detection model and are more often wrong.

A photo is correctly annotated when every fish and every free-moving marine animal larger than \
about 3 cm has one box of the right class, even if only part of it is visible (a tail, a fin), \
each box fits its animal tightly (small gaps are fine), and nothing else has a box.
""" + CLASSES + """

Review every box and give a verdict, with how sure you are of it ("high", "medium" or "low"):
- "remove": an extra box, on something that should not have one, or a second box on an animal \
that already has one.
- "fix": a box that is clearly too big, too small, cut off or shifted (not for small gaps), a box \
that covers two animals (fix it to cover one of them and add the other under "missing"), or a \
box with the wrong class. Give the corrected coordinates and class.
- "keep": anything else; repeat its coordinates and class unchanged.
Then list under "missing" every animal in this {subject} that has no box: check the edges, the \
background and behind the reef. Give its class, tight coordinates and a confidence.
{unusable_note}

Boxes:
{boxes}"""
UNUSABLE_NOTE = """If the photo is too blurry or dark to judge, is corrupted or is not underwater, set \
"unusable" to true; then nothing else needs marking."""
TILE_UNUSABLE_NOTE = """Review every box even if this crop is murky or dark: the photo as a whole is \
usable, so set "unusable" to false."""

BOX_FIELDS = {
    "class": {"type": "string", "enum": ["fish", "non-fish"]},
    "x_min": {"type": "integer"},
    "y_min": {"type": "integer"},
    "x_max": {"type": "integer"},
    "y_max": {"type": "integer"},
}
CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}


def schema(version: int) -> dict:
    review_fields = {"id": {"type": "integer"}, "verdict": {"type": "string", "enum": ["keep", "remove", "fix"]},
                     **BOX_FIELDS}
    if version >= 2:
        review_fields["confidence"] = CONFIDENCE
    return {
        "type": "object",
        "properties": {
            "unusable": {"type": "boolean"},
            "reviews": {
                "type": "array",
                "items": {"type": "object", "properties": review_fields, "required": list(review_fields),
                          "additionalProperties": False},
            },
            "missing": {
                "type": "array",
                "items": {"type": "object", "properties": {**BOX_FIELDS, "confidence": CONFIDENCE},
                          "required": [*BOX_FIELDS, "confidence"], "additionalProperties": False},
            },
        },
        "required": ["unusable", "reviews", "missing"],
        "additionalProperties": False,
    }


def input_boxes(file_name: str) -> list[dict]:
    """Existing dataset boxes, then the first pass's boxes that match no existing box."""
    existing = pilot.existing_boxes_by_file().get(file_name, [])
    boxes = [{"box": b, "class": "fish", "source": "existing"} for b in existing]
    first = json.loads(pilot.raw_path(FIRST_PASS, file_name).read_text())["boxes"]
    fish = [b for b in first if b.get("class", "fish") == "fish"]
    matches = pilot.match([b["box"] for b in fish], existing)
    added = [b for i, b in enumerate(fish) if i not in matches]
    added += [b for b in first if b.get("class", "fish") == "non-fish"]
    boxes += [{"box": b["box"], "class": b.get("class", "fish"), "source": f"first pass ({b['confidence']})"}
              for b in added]
    return boxes


def draw_numbered(photo: Image.Image, boxes: list[dict], crop=None, upscale=UPSCALE, colored=False):
    """The photo (or a crop of it) upscaled, with boxes drawn and numbered 1..n in list order."""
    crop = crop or (0, 0, photo.width, photo.height)
    region = photo.crop(crop)
    image = region.resize((region.width * upscale, region.height * upscale), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    for n, b in enumerate(boxes, 1):
        x0, y0, x1, y1 = ((v - crop[i % 2]) * upscale for i, v in enumerate(b["box"]))
        color = (0, 230, 0) if colored and b["source"] == "existing" else (255, 255, 0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.rectangle([x0, y0, x0 + 8 * len(str(n)) + 4, y0 + 13], fill=(0, 0, 0))
        draw.text((x0 + 2, y0), str(n), fill=color)
    return image


def to_photo(r: dict, crop, upscale, width, height) -> list[float]:
    x0, x1 = sorted((r["x_min"] / upscale + crop[0], r["x_max"] / upscale + crop[0]))
    y0, y1 = sorted((r["y_min"] / upscale + crop[1], r["y_max"] / upscale + crop[1]))
    return [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]


def listing(boxes: list[dict], crop, upscale, with_source: bool) -> str:
    lines = []
    for n, b in enumerate(boxes, 1):
        x0 = max(0, round((b["box"][0] - crop[0]) * upscale))
        y0 = max(0, round((b["box"][1] - crop[1]) * upscale))
        x1 = min(round((crop[2] - crop[0]) * upscale), round((b["box"][2] - crop[0]) * upscale))
        y1 = min(round((crop[3] - crop[1]) * upscale), round((b["box"][3] - crop[1]) * upscale))
        source = (" (green, annotated by a person)" if b["source"] == "existing" else " (yellow, proposed by a model)")
        lines.append(f"{n}: {b['class']} [{x0}, {y0}, {x1}, {y1}]" + (source if with_source else ""))
    return "\n".join(lines) or "(none)"


def review_jobs(reviewer: str, width: int, height: int, boxes: list[dict]) -> list[dict]:
    """The requests of one photo: the whole photo, or (v2, crowded photos) 4 overlapping tiles."""
    config = REVIEWERS[reviewer]
    if config["version"] >= 2 and len(boxes) > TILE_THRESHOLD:
        return [{"crop": crop, "quadrant": quadrant, "upscale": pilot.TILE_UPSCALE,
                 "visible": [i for i, b in enumerate(boxes) if pilot.overlap(b["box"], list(crop)) > 0.5]}
                for crop, quadrant in pilot.tiles(width, height)]
    return [{"crop": (0, 0, width, height), "quadrant": None, "upscale": UPSCALE, "visible": list(range(len(boxes)))}]


def review_request(reviewer: str, photo: Image.Image, boxes: list[dict], job: dict) -> dict:
    config = REVIEWERS[reviewer]
    crop, quadrant, upscale = job["crop"], job["quadrant"], job["upscale"]
    local = [boxes[i] for i in job["visible"]]
    image = draw_numbered(photo, local, crop, upscale, colored=config["version"] >= 2)
    if config["version"] >= 2:
        prompt = PROMPT_V2.format(
            subject="image" if quadrant else "photo", width=image.width, height=image.height,
            crop_note=" It is an enlarged crop of a photo." if quadrant else "", n=len(local),
            boxes=listing(local, crop, upscale, with_source=True),
            unusable_note=TILE_UNUSABLE_NOTE if quadrant else UNUSABLE_NOTE)
    else:
        prompt = PROMPT.format(width=image.width, height=image.height, n=len(local),
                               boxes=listing(local, crop, upscale, with_source=False))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    data = base64.standard_b64encode(buffer.getvalue()).decode()
    prompts = {"system": SYSTEM_PROMPT, "schema": schema(config["version"])}
    return pilot.request_params(config["model"], config["effort"], prompts, data, prompt)


def needs_retry(job: dict, result: dict | None) -> bool:
    """The model sometimes skips the review list of a crowded image: such an answer is asked once more."""
    return result is not None and bool(job["visible"]) and len(result["reviews"]) < len(job["visible"]) / 2


def review_merge(reviewer: str, file_name: str, width: int, height: int, boxes: list[dict], jobs: list[dict],
                 outcomes: list[list]) -> dict:
    """
    Record of one photo from the attempts [(call, answer), ...] of each of its requests, in job
    order. Of several successful attempts, the one with the most box reviews is used.
    """
    config = REVIEWERS[reviewer]
    record = {"reviewer": reviewer, **config, "first_pass": FIRST_PASS, "file_name": file_name,
              "input_boxes": boxes, "calls": [], "mode": "tiled" if jobs[0]["quadrant"] else "full"}
    reviews, missing, unusable_votes = {}, [], []
    for job, attempts in zip(jobs, outcomes):
        crop, quadrant, upscale, visible = job["crop"], job["quadrant"], job["upscale"], job["visible"]
        result = None
        for n, (call, answer) in enumerate(attempts):
            call["crop"] = list(crop)
            if n:
                call["retry"] = True
            record["calls"].append(call)
            if answer is not None and (result is None or len(answer["reviews"]) > len(result["reviews"])):
                result = answer
        if result is None:
            record["error"] = attempts[-1][0].get("error", "no answer") if attempts else "no answer"
            break
        unusable_votes.append(result["unusable"])

        def owned(box):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            return quadrant is None or (quadrant[0] <= cx < quadrant[2] and quadrant[1] <= cy < quadrant[3])

        for r in result["reviews"]:
            if not 1 <= r["id"] <= len(visible):
                continue
            global_id = visible[r["id"] - 1] + 1
            original = boxes[global_id - 1]
            if not owned(original["box"]):
                continue
            review = {"id": global_id, "verdict": r["verdict"], "class": r["class"],
                      "box": to_photo(r, crop, upscale, width, height), "confidence": r.get("confidence")}
            inside = pilot.overlap(original["box"], list(crop)) > 0.99
            if review["verdict"] == "fix" and not inside:
                review["box"] = original["box"]  # a box cut by the tile edge keeps its coordinates
            if (config["version"] >= 2 and review["verdict"] == "remove" and original["source"] == "existing"
                    and review["confidence"] != "high"):
                review["verdict"], review["overruled"] = "keep", True
                review["box"] = original["box"]
            reviews[global_id] = review
        for m in result["missing"]:
            box = to_photo(m, crop, upscale, width, height)
            if owned(box) and box[2] - box[0] > 1 and box[3] - box[1] > 1:
                missing.append({"class": m["class"], "confidence": m["confidence"], "box": box})
    if len(outcomes) < len(jobs) and "error" not in record:
        record["error"] = "missing results"

    usage = {k: sum(c.get("usage", {}).get(k, 0) for c in record["calls"]) for k in ("input_tokens", "output_tokens")}
    record["usage"] = usage
    record["elapsed_s"] = round(sum(c.get("elapsed_s", 0) for c in record["calls"]), 1)
    if "error" in record:
        return record
    # A photo crowded enough to be tiled is usable; a single tile's vote is ignored
    record["unusable"] = any(unusable_votes) if record["mode"] == "full" else False
    record["reviews"] = [reviews[k] for k in sorted(reviews)]
    if config["version"] >= 2:
        # Drop added animals that duplicate a box that stays, or another added animal
        kept = [r["box"] for r in record["reviews"] if r["verdict"] != "remove"]
        kept += [b["box"] for n, b in enumerate(boxes, 1) if n not in reviews]  # not reviewed: kept as is
        deduped = []
        for m in pilot.suppress_duplicates(missing):
            if all(pilot.iou(m["box"], k) < pilot.NMS_IOU and pilot.overlap(m["box"], k) < pilot.NMS_OVERLAP
                   for k in kept):
                deduped.append(m)
        missing = deduped
    record["missing"] = missing
    return record


def review_photo(client: anthropic.Anthropic, reviewer: str, file_name: str) -> dict:
    boxes = input_boxes(file_name)
    with Image.open(pilot.IMAGES_DIR / file_name) as photo:
        photo = photo.convert("RGB")
        width, height = photo.size
        jobs = review_jobs(reviewer, width, height, boxes)
        outcomes = []
        for job in jobs:
            params = review_request(reviewer, photo, boxes, job)
            attempts = [pilot.ask(client, params, None)]
            if attempts[0][1] is None:
                outcomes.append(attempts)
                break
            if needs_retry(job, attempts[0][1]):
                attempts.append(pilot.ask(client, params, None))
            outcomes.append(attempts)
    return review_merge(reviewer, file_name, width, height, boxes, jobs, outcomes)


def raw_path(reviewer: str, file_name: str) -> Path:
    return pilot.PILOT_DIR / "raw" / f"{reviewer}__{FIRST_PASS}" / f"{Path(file_name).stem}.json"


def run(args):
    pilot.load_env_file()
    client = anthropic.Anthropic()
    jobs = [(r, i["file_name"]) for r in args.reviewers for i in pilot.pilot_images()
            if not raw_path(r, i["file_name"]).exists()]
    print(f"{len(jobs)} photos to review")

    def job(reviewer_and_file):
        reviewer, file_name = reviewer_and_file
        record = review_photo(client, reviewer, file_name)
        if "error" in record and not record["usage"]["input_tokens"]:
            print(f"  {reviewer} {file_name[16:31]}: {record['error']} (not cached)")
            return
        path = raw_path(reviewer, file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        if "error" in record:
            print(f"  {reviewer} {file_name[16:31]}: {record['error']}")
            return
        verdicts = [r["verdict"] for r in record["reviews"]]
        overruled = sum(bool(r.get("overruled")) for r in record["reviews"])
        print(f"  {reviewer} {file_name[16:31]} ({record['mode']}): {len(record['input_boxes'])} boxes -> keep "
              f"{verdicts.count('keep')}, remove {verdicts.count('remove')}, fix {verdicts.count('fix')}, "
              f"missing +{len(record['missing'])}, overruled removals {overruled}, unusable {record['unusable']}, "
              f"{record['elapsed_s']}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, jobs))


def render(args):
    """Before (numbered input boxes) and after (kept green, fixed blue, removed red X, added magenta)."""
    out_dir = pilot.PILOT_DIR / args.sheets_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in args.reviewers:
        cost = 0.0
        price_in, price_out = pilot.PRICES[REVIEWERS[reviewer]["model"]]
        colored = REVIEWERS[reviewer]["version"] >= 2
        for image in pilot.pilot_images():
            record = json.loads(raw_path(reviewer, image["file_name"]).read_text())
            cost += (record["usage"]["input_tokens"] * price_in + record["usage"]["output_tokens"] * price_out) / 1e6
            photo = Image.open(pilot.IMAGES_DIR / image["file_name"]).convert("RGB")
            before = cv2.cvtColor(np.array(draw_numbered(photo, record["input_boxes"], colored=colored)),
                                  cv2.COLOR_RGB2BGR)
            after = cv2.resize(cv2.imread(str(pilot.IMAGES_DIR / image["file_name"])), None, fx=UPSCALE, fy=UPSCALE,
                               interpolation=cv2.INTER_CUBIC)
            s = UPSCALE
            reviews = {r["id"]: r for r in record.get("reviews", [])}
            for n, b in enumerate(record["input_boxes"], 1):
                r = reviews.get(n)
                x0, y0, x1, y1 = (int(v * s) for v in b["box"])
                if r is None:
                    cv2.rectangle(after, (x0, y0), (x1, y1), (128, 128, 128), 1)
                elif r["verdict"] == "remove":
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.line(after, (x0, y0), (x1, y1), (0, 0, 255), 2)
                elif r["verdict"] == "fix":
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 0, 255), 1)
                    fx0, fy0, fx1, fy1 = (int(v * s) for v in r["box"])
                    cv2.rectangle(after, (fx0, fy0), (fx1, fy1), (255, 128, 0), 2)
                else:
                    cv2.rectangle(after, (x0, y0), (x1, y1), (0, 220, 0), 2)
                if r is not None and r["class"] == "non-fish":
                    cv2.putText(after, "NF", (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for m in record.get("missing", []):
                x0, y0, x1, y1 = (int(v * s) for v in m["box"])
                cv2.rectangle(after, (x0, y0), (x1, y1), (255, 0, 255), 2)
            verdicts = [r["verdict"] for r in record.get("reviews", [])]
            overruled = sum(bool(r.get("overruled")) for r in record.get("reviews", []))
            labels = [f"before: {len(record['input_boxes'])} boxes" + (" (green = people, yellow = first pass)"
                                                                         if colored else " (existing + first pass)"),
                      f"{reviewer} ({record.get('mode', 'full')}): keep {verdicts.count('keep')}, fix "
                      f"{verdicts.count('fix')} (blue), remove {verdicts.count('remove')} (red X), "
                      f"+{len(record.get('missing', []))} (magenta)" + (f", {overruled} removals overruled"
                                                                         if overruled else "")]
            panels = []
            for panel, label in zip((before, after), labels):
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                panels.append(cv2.resize(panel, (960, 720)))
            cv2.imwrite(str(out_dir / f"pass2_{reviewer}_{int(image['audit_id']):02d}.jpg"), np.hstack(panels),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{reviewer}: sheets written to {out_dir}; review cost on the pilot photos: ${cost:.3f}")


def final_annotation(record: dict, policy: str = "protect") -> list[dict]:
    """
    The photo's boxes after both passes: what the review kept or fixed, plus what it added.
    policy "review": every verdict of the review is applied, as run.
    policy "protect": the existing (human) boxes stay "fish", and the review may remove a
        first-pass box only if the first pass gave it low confidence (in the 50-photo test most
        of the high/medium first-pass boxes the review removed were real fish).
    """
    if record.get("unusable"):
        return []
    reviews = {r["id"]: r for r in record["reviews"]}
    boxes = []
    for n, b in enumerate(record["input_boxes"], 1):
        r = reviews.get(n)
        source = "existing" if b["source"] == "existing" else "first pass"
        confidence = b["source"].split("(")[1][:-1] if source == "first pass" else None
        if r is not None and policy == "protect":
            if r["verdict"] == "remove" and source == "first pass" and confidence != "low":
                r = {**r, "verdict": "keep", "box": b["box"], "class": b["class"]}
            if source == "existing" and r["class"] != "fish":
                r = {**r, "class": "fish"}
        if r is None or r["verdict"] == "keep":
            # A kept box stays exactly as it was (the review's repeated coordinates are approximate,
            # and cut at the tile edge in tiled reviews)
            boxes.append({"box": b["box"], "class": b["class"], "source": source, "status": "kept", "confidence": confidence})
        elif r["verdict"] != "remove":
            status = "fixed" if r["verdict"] == "fix" else "kept"
            boxes.append({"box": r["box"], "class": r["class"], "source": source, "status": status,
                          "confidence": confidence, **({"original_box": b["box"]} if status == "fixed" else {})})
    boxes += [{"box": m["box"], "class": m["class"], "source": "review", "status": "added",
               "confidence": m["confidence"]} for m in record["missing"]]
    return boxes


def rejected_candidates(record: dict, policy: str = "protect") -> list[dict]:
    """Boxes that final_annotation leaves out because the review removed them (human or first pass)."""
    if record.get("unusable"):
        return []
    reviews = {r["id"]: r for r in record["reviews"]}
    rejected = []
    for n, b in enumerate(record["input_boxes"], 1):
        r = reviews.get(n)
        if r is None or r["verdict"] != "remove":
            continue
        if b["source"] == "existing":
            rejected.append({"box": b["box"], "class": b["class"], "source": "human", "confidence": None,
                             "review_confidence": r.get("confidence")})
            continue
        confidence = b["source"].split("(")[1][:-1]
        if policy == "review" or confidence == "low":
            rejected.append({"box": b["box"], "class": b["class"], "source": "vlm_pass1", "confidence": confidence,
                             "review_confidence": r.get("confidence")})
    return rejected


def final(args):
    """Final annotations of the listed photos, their statistics and measured cost, and before/after sheets."""
    from collections import Counter

    reviewer = args.reviewers[0]
    out_dir = pilot.PILOT_DIR / args.sheets_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    images = pilot.pilot_images()
    existing = pilot.existing_boxes_by_file()
    stats, cost, results = Counter(), Counter(), {}
    for image in images:
        first = json.loads(pilot.raw_path(FIRST_PASS, image["file_name"]).read_text())
        record = json.loads(raw_path(reviewer, image["file_name"]).read_text())
        for name, rec, model in (("pass 1", first, pilot.VARIANTS[FIRST_PASS]["model"]),
                                 ("pass 2", record, REVIEWERS[reviewer]["model"])):
            price_in, price_out = pilot.PRICES[model]
            cost[name] += (rec["usage"]["input_tokens"] * price_in + rec["usage"]["output_tokens"] * price_out) / 1e6
        boxes = final_annotation(record, args.policy)
        results[image["file_name"]] = boxes
        stats["photos"] += 1
        stats["unusable photos"] += bool(record.get("unusable"))
        stats["existing boxes"] += len(existing.get(image["file_name"], []))
        stats["final boxes"] += len(boxes)
        stats["final non-fish boxes"] += sum(b["class"] == "non-fish" for b in boxes)
        for b in boxes:
            stats[f"final: {b['source']} {b['status']}" + (f" ({b['confidence']})" if b["source"] == "first pass" else "")] += 1
        reviewed = {r["id"]: r for r in record["reviews"]}
        stats["removed: first pass"] += sum(1 for n, b in enumerate(record["input_boxes"], 1)
                                            if b["source"] != "existing" and reviewed.get(n, {}).get("verdict") == "remove"
                                            and (args.policy == "review" or "(low)" in b["source"]))
        stats["removed: existing"] += sum(1 for n, b in enumerate(record["input_boxes"], 1)
                                          if b["source"] == "existing" and reviewed.get(n, {}).get("verdict") == "remove")
    n = stats["photos"]
    print(f"{n} photos, reviewer {reviewer}, first pass {FIRST_PASS}, policy {args.policy}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    total = cost["pass 1"] + cost["pass 2"]
    print(f"measured cost: pass 1 ${cost['pass 1']:.4f} + pass 2 ${cost['pass 2']:.4f} = ${total:.4f} "
          f"(${total / n:.5f} per photo)")
    print(f"for all {pilot.DATASET_IMAGES} photos: ${total / n * pilot.DATASET_IMAGES:.1f} standard, "
          f"${total / n * pilot.DATASET_IMAGES / 2:.1f} with the Batch API")
    (out_dir / "final_annotations.json").write_text(json.dumps(results, indent=1))

    colors = {"existing kept": (0, 220, 0), "existing fixed": (255, 128, 0), "review added": (0, 140, 255)}
    rows = []
    for image in images:
        photo = cv2.imread(str(pilot.IMAGES_DIR / image["file_name"]))
        scale = 1.25
        before = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        after = before.copy()
        for b in existing.get(image["file_name"], []):
            cv2.rectangle(before, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)),
                          (0, 220, 0), 2)
        for b in results[image["file_name"]]:
            key = f"{b['source']} {b['status']}"
            color = colors.get(key, (255, 0, 255) if b["confidence"] in ("high", "medium") else (0, 255, 255))
            if b["class"] == "non-fish":
                color = (0, 0, 255)
            x0, y0, x1, y1 = (int(v * scale) for v in b["box"])
            cv2.rectangle(after, (x0, y0), (x1, y1), color, 2)
        for panel, label in ((before, f"#{image['audit_id']} before: {len(existing.get(image['file_name'], []))} boxes"),
                             (after, f"after: {len(results[image['file_name']])} boxes")):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(panel, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(np.hstack([before, after]))
    legend = ("after: green = kept human box, blue = tightened human box, magenta = added (high/medium), "
              "yellow = added (low), orange = added by review, red = non-fish")
    for i in range(0, len(rows), 3):
        sheet = np.vstack(rows[i:i + 3])
        head = np.full((30, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(head, legend, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"final_{i // 3 + 1:02d}.jpg"), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"before/after sheets written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "render", "final"])
    parser.add_argument("--reviewers", nargs="+", default=list(REVIEWERS), choices=list(REVIEWERS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sheets-dir", default="sheets_pass2")
    parser.add_argument("--policy", choices=["protect", "review"], default="protect",
                        help="final: how review verdicts are applied (see final_annotation)")
    parser.add_argument("--images", default=str(pilot.IMAGE_LIST), help="TSV list of photos (default: the 9 pilot photos)")
    args = parser.parse_args()
    pilot.IMAGE_LIST = Path(args.images)
    {"run": run, "render": render, "final": final}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
