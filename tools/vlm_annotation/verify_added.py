"""
Clean the boxes the two passes added to a dataset (batch_run.py export), before they are added to it.
The dataset is the config named by VLM_DATASET (see pilot.py); its "verify" section describes the
photos for the check (see configs/README.md).

1. static: boxes the models added at almost the same place (IoU >= STATIC_IOU) in STATIC_FRAMES or more
   frames of one video are marked static: algae, debris and cable bends stay put. Animals can also sit
   still for minutes, and in busy videos animals overlap by chance, so static boxes are not dropped
   automatically; every static group is written out for a visual check. The video of a frame is the
   first group of the config's verify.video_pattern regex on its file name (no pattern: one video,
   as for a fixed camera).
2. submit / collect: every added box of medium or low confidence (static or not) is checked by
   Claude Opus 5.5 on the Message Batches API. Each request shows an enlarged crop with the box drawn
   and the whole frame with the box marked, and asks what is inside the box: fish, non-fish or no
   animal, with what it is and how sure the model is. `test` sends a few of them directly first.
3. select: the boxes to add to the dataset, from the verdicts and a decisions file written after
   the visual checks: boxes on an animal that already has a human box, boxes in static groups judged
   to be objects and boxes in excluded videos are dropped; high-confidence boxes are kept (with class
   fixes); medium and low confidence boxes are kept if Opus called them fish or non-fish with high or
   medium confidence, with Opus's class. Decisions file:
     {"object_groups": [indices in static_groups.json judged to be objects],
      "animal_groups": [indices judged to be animals (not used, for the record)],
      "excluded_groups": [file-name parts, e.g. video names, whose added boxes are all dropped],
      "class_fixes": {"a<annotation id>": "fish" or "non-fish", for high-confidence boxes}}
   The output is keyed by file name without the config's prefix, as dataset scripts read their source.

Usage (from the repo root, ANTHROPIC_API_KEY in .env):
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py static
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py test -n 3
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py submit
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py collect
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/verify_added.py select \
      --decisions decisions.json --out datasets/<name>_added_boxes.json
Writes to <run dir>/verify/: static_groups.json, jobs.json, state.json, results.jsonl, verdicts.json.
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import pilot  # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
PRICE_IN, PRICE_OUT = 4.0, 20.0  # USD per million tokens, before the Batch API discount
STATIC_IOU = 0.5
STATIC_FRAMES = 3
CROP_SIDE = 448
FRAME_WIDTH = 960
VERIFY = pilot.DATASET_CONFIG["verify"]
VIDEO = re.compile(VERIFY["video_pattern"]) if VERIFY.get("video_pattern") else None

SYSTEM = "You check bounding boxes proposed by a detection model in underwater photos, for the training set of a fish detector."
PROMPT = """{setting}

Image 1 is an enlarged crop around a red box; image 2 is the whole frame with the same box in red.
What is inside the red box? Answer with one label:
- "fish": a fish ({fish_examples}), even if only part of it is visible or it is cut by the box \
or the frame edge.
- "non-fish": another free-moving animal clearly bigger than about 3 cm ({non_fish_examples}).
- "no animal": {no_animal}.
- "unclear": you cannot tell.
Also say in a few words what it is, and how sure you are ("high", "medium" or "low").""".format(
    setting=VERIFY.get("setting") or pilot.SETTING.replace("It comes", "The photos come", 1),
    fish_examples=VERIFY.get("fish_examples", "of any kind"),
    non_fish_examples=VERIFY.get("non_fish_examples", "crab, lobster, shrimp, octopus, squid, starfish, sea urchin, jellyfish..."),
    no_animal=VERIFY.get("no_animal", "algae, debris, particles, a cable or rope, equipment, a shadow, sand or rock, or "
                                      "anything else that does not move on its own"),
)
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["fish", "non-fish", "no animal", "unclear"]},
        "what": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["label", "what", "confidence"],
    "additionalProperties": False,
}


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union > 0 else 0.0


def added_boxes(run_dir: Path) -> list[dict]:
    coco = json.loads((run_dir / "export" / f"{pilot.DATASET}_vlm_coco.json").read_text())
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    boxes = []
    for a in coco["annotations"]:
        if a["attributes"]["source"] == "human":
            continue
        boxes.append({"key": f"a{a['id']}", "annotation_id": a["id"], "file_name": names[a["image_id"]],
                      "bbox": a["bbox"], "class": "fish" if a["category_id"] == 1 else "non-fish",
                      "source": a["attributes"]["source"], "confidence": a["attributes"]["confidence"]})
    return boxes


def static_groups(boxes: list[dict]) -> list[list[dict]]:
    """Groups (union-find) of added boxes at the same place in different frames of one video."""
    by_video = defaultdict(list)
    for b in boxes:
        found = VIDEO.match(b["file_name"]) if VIDEO else None
        by_video[found[1] if found else ("all" if VIDEO is None else b["file_name"])].append(b)
    groups = []
    for items in by_video.values():
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i]["file_name"] != items[j]["file_name"] and iou(items[i]["bbox"], items[j]["bbox"]) >= STATIC_IOU:
                    parent[find(i)] = find(j)
        members = defaultdict(list)
        for i, b in enumerate(items):
            members[find(i)].append(b)
        groups += [g for g in members.values() if len({b["file_name"] for b in g}) >= STATIC_FRAMES]
    return groups


def overlap(a, b):
    """Intersection over the smaller box's area, for [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return ix * iy / smaller if smaller > 0 else 0.0


def select(run_dir: Path, decisions_path: Path, out_path: Path):
    """The added boxes to keep, as {file name: [{bbox, class, source, confidence, verified}]}."""
    decisions = json.loads(decisions_path.read_text())
    groups = json.loads((run_dir / "verify" / "static_groups.json").read_text())
    verdicts = json.loads((run_dir / "verify" / "verdicts.json").read_text())
    objects = {b["key"] for i in decisions["object_groups"] for b in groups[i]}
    coco = json.loads((run_dir / "export" / f"{pilot.DATASET}_vlm_coco.json").read_text())
    human = defaultdict(list)
    for a in coco["annotations"]:
        if a["attributes"]["source"] == "human":
            human[a["image_id"]].append(a["bbox"])
    image_ids = {image["file_name"]: image["id"] for image in coco["images"]}
    excluded = decisions.get("excluded_groups", decisions.get("excluded_recordings", []))
    kept, reasons = defaultdict(list), defaultdict(int)
    for b in added_boxes(run_dir):
        if any(iou(b["bbox"], h) >= pilot.MATCH_IOU or overlap(b["bbox"], h) >= pilot.MATCH_OVERLAP
               for h in human[image_ids[b["file_name"]]]):
            reasons["on a human box"] += 1
            continue
        if b["key"] in objects:
            reasons["static object"] += 1
            continue
        if any(group in b["file_name"] for group in excluded):
            reasons["excluded video"] += 1
            continue
        entry = {"bbox": [round(v, 1) for v in b["bbox"]], "class": b["class"], "source": b["source"],
                 "confidence": b["confidence"]}
        if b["confidence"] == "high":
            entry["class"] = decisions["class_fixes"].get(b["key"], b["class"])
        else:
            v = verdicts.get(b["key"])
            if not v or v["label"] not in ("fish", "non-fish") or v["confidence"] not in ("high", "medium"):
                reasons[f"{b['confidence']} confidence, not confirmed by Opus"] += 1
                continue
            entry["class"] = v["label"]
            entry["verified"] = f"{v['label']} ({v['confidence']})"
        kept[b["file_name"].removeprefix(pilot.DATASET_CONFIG["prefix"])].append(entry)
    for entries in kept.values():
        entries.sort(key=lambda e: (e["bbox"][0], e["bbox"][1]))
    lines = [f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in sorted(kept.items())]
    out_path.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"kept {sum(len(v) for v in kept.values())} boxes in {len(kept)} frames; dropped {dict(reasons)}")


def images_for(box: dict) -> tuple[str, str]:
    """Base64 JPEGs: the enlarged crop with the box, and the whole frame with the box."""
    with Image.open(pilot.IMAGES_DIR / box["file_name"]) as image:
        image = image.convert("RGB")
    W, H = image.size
    x, y, w, h = box["bbox"]
    pad = max(1.0 * max(w, h), 30)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    crop = image.crop((int(x0), int(y0), int(x1 + 0.999), int(y1 + 0.999)))
    s = CROP_SIDE / max(crop.size)
    crop = crop.resize((round(crop.size[0] * s), round(crop.size[1] * s)), Image.BICUBIC)
    ImageDraw.Draw(crop).rectangle([(x - int(x0)) * s, (y - int(y0)) * s, (x + w - int(x0)) * s, (y + h - int(y0)) * s],
                                   outline=(255, 0, 0), width=2)
    f = FRAME_WIDTH / W
    frame = image.resize((FRAME_WIDTH, round(H * f)), Image.BICUBIC)
    ImageDraw.Draw(frame).rectangle([(x - 3) * f, (y - 3) * f, (x + w + 3) * f, (y + h + 3) * f], outline=(255, 0, 0), width=3)
    out = []
    for img in (crop, frame):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out[0], out[1]


def params(box: dict) -> dict:
    crop, frame = images_for(box)
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Image 1:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop}},
            {"type": "text", "text": "Image 2:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}},
            {"type": "text", "text": PROMPT},
        ]}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["static", "submit", "collect", "test", "select"])
    parser.add_argument("--run-dir", type=Path, default=Path("data/vlm_annotation") / pilot.DATASET)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--decisions", type=Path, help="select: decisions file of the visual checks")
    parser.add_argument("--out", type=Path, help="select: output JSON of the boxes to add")
    args = parser.parse_args()
    out = args.run_dir / "verify"
    out.mkdir(parents=True, exist_ok=True)
    pilot.load_env_file()

    if args.command == "select":
        select(args.run_dir, args.decisions, args.out)
        return

    if args.command == "static":
        boxes = added_boxes(args.run_dir)
        groups = static_groups(boxes)
        (out / "static_groups.json").write_text(json.dumps(groups, indent=1))
        print(f"{len(boxes)} added boxes; {len(groups)} static groups with "
              f"{sum(len(g) for g in groups)} boxes written to {out / 'static_groups.json'}")
        return

    boxes = added_boxes(args.run_dir)
    static = {b["key"] for g in json.loads((out / "static_groups.json").read_text()) for b in g}
    # Static boxes are checked too: crabs and gobies also sit still for minutes (see the visual check)
    jobs = [{**b, "static": b["key"] in static} for b in boxes if b["confidence"] in ("medium", "low")]
    client = anthropic.Anthropic()
    state_path = out / "state.json"

    if args.command == "test":
        for box in jobs[: args.n]:
            m = client.messages.create(**params(box))
            text = next(c.text for c in m.content if c.type == "text")
            print(box["file_name"], box["class"], box["confidence"], text, m.usage.input_tokens, m.usage.output_tokens)
        return

    if args.command == "submit":
        if state_path.exists():
            raise SystemExit("already submitted")
        requests = [{"custom_id": box["key"], "params": params(box)} for box in jobs]
        batch = client.messages.batches.create(requests=requests)
        (out / "jobs.json").write_text(json.dumps(jobs))
        state_path.write_text(json.dumps({"batch": batch.id, "requests": len(requests)}))
        print(f"submitted {batch.id} with {len(requests)} requests ({sum(j['static'] for j in jobs)} of them static)")
        return

    state = json.loads(state_path.read_text())
    while True:
        info = client.messages.batches.retrieve(state["batch"])
        print(time.strftime("%H:%M:%S"), info.processing_status, info.request_counts.to_dict(), flush=True)
        if info.processing_status == "ended":
            break
        time.sleep(60)
    verdicts, tokens = {}, {"input": 0, "output": 0}
    with open(out / "results.jsonl", "w") as f:
        for r in client.messages.batches.results(state["batch"]):
            row = {"key": r.custom_id, "type": r.result.type}
            if r.result.type == "succeeded":
                m = r.result.message
                tokens["input"] += m.usage.input_tokens
                tokens["output"] += m.usage.output_tokens
                row["stop"] = m.stop_reason
                try:
                    row.update(json.loads(next(c.text for c in m.content if c.type == "text")))
                    verdicts[r.custom_id] = {k: row[k] for k in ("label", "what", "confidence")}
                except Exception as e:  # noqa: BLE001
                    row["error"] = str(e)
            else:
                row["error"] = str(r.result)
            f.write(json.dumps(row) + "\n")
    usd = (tokens["input"] * PRICE_IN + tokens["output"] * PRICE_OUT) / 1e6 * 0.5
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    (out / "cost.json").write_text(json.dumps({**tokens, "usd_batch": round(usd, 3), "verdicts": len(verdicts)}, indent=1))
    print(f"{len(verdicts)} verdicts of {state['requests']}, tokens {tokens}, cost ${usd:.2f}")


if __name__ == "__main__":
    main()
