"""
Export of the two-pass VLM pre-annotation (batch_run.py) for the human annotators.

Writes to <run dir>/export/:
  obsea_2013_2014_vlm_coco.json  COCO with the photos and their boxes after both passes (categories
      fish = 1, non-fish = 2, as in the rest of the pipeline). Every annotation has "attributes":
        source       human | vlm_pass1 | vlm_review (who drew the box)
        change       unchanged | box_corrected (a human box tightened by the review) | added
        confidence   high | medium | low (boxes drawn by a model only)
        original_bbox  the box before the review corrected it (box_corrected only)
      and every image has a "vlm" object: status (reviewed, pass1_only if the review failed,
      human_only if pass 1 failed, damaged_file if the photo file cannot be decoded completely;
      such photos are also unusable), unusable (the review found the photo too blurry or dark;
      its human boxes are then kept as they are and nothing is added), crowded (reviewed in
      tiles), check_priority and box counts.
  rejected_candidates.json  {file name: [boxes]}: boxes the review removed, kept here for the
      annotators: low-confidence pass-1 boxes (about half of them were real fish in the pilot)
      and the rare human boxes removed on a high-confidence verdict.
  review_queue.csv  photos sorted by check_priority, highest first.
  summary.json  totals.

check_priority estimates how much a photo is likely to need a human: one point per uncertain
box (low-confidence model boxes, medium-confidence boxes added by the review, non-fish boxes,
rejected candidates), plus 3 if crowded (crowded photos still miss the most fish) and 5 if the photo is
unusable or was not reviewed. A photo at 0 has only human boxes (unchanged or tightened) and
model boxes of high or medium confidence.

Usage (from the repo root): venv-vlm/bin/python tools/vlm_annotation/export.py [--run-dir DIR]
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pilot  # noqa: E402
import second_pass  # noqa: E402
from datasets.settings import Settings  # noqa: E402

PASS1 = second_pass.FIRST_PASS
REVIEWER = "review_haiku_v2"
POLICY = "protect"
SOURCES = {"existing": "human", "first pass": "vlm_pass1", "review": "vlm_review"}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return None if "error" in record else record


def photo_boxes(photo: dict, damaged: bool = False) -> tuple[list[dict], list[dict], dict]:
    """Final boxes, rejected candidates and image-level info of one photo."""
    file_name = photo["file_name"]
    review = load(second_pass.raw_path(REVIEWER, file_name))
    existing = pilot.existing_boxes_by_file().get(file_name, [])
    classes = pilot.existing_classes_by_file().get(file_name, ["fish"] * len(existing))
    human = [{"box": b, "class": c, "source": "existing", "status": "kept", "confidence": None}
             for b, c in zip(existing, classes)]
    info = {"status": "damaged_file" if damaged else "human_only", "unusable": damaged, "crowded": False}
    if damaged:
        return human, [], info
    if review is not None:
        info.update(status="reviewed", unusable=bool(review.get("unusable")), crowded=review["mode"] == "tiled")
        if info["unusable"]:
            return human, [], info
        return (second_pass.final_annotation(review, POLICY), second_pass.rejected_candidates(review, POLICY), info)
    if load(pilot.raw_path(PASS1, file_name)) is not None:
        info["status"] = "pass1_only"
        boxes = []
        for b in second_pass.input_boxes(file_name):
            if b["source"] == "existing":
                boxes.append({**b, "status": "kept", "confidence": None})
            else:
                boxes.append({**b, "source": "first pass", "status": "kept", "confidence": b["source"].split("(")[1][:-1]})
        return boxes, [], info
    return human, [], info


def xywh(box: list[float]) -> list[float]:
    return [round(box[0], 2), round(box[1], 2), round(box[2] - box[0], 2), round(box[3] - box[1], 2)]


def export(run_dir: Path, photos: list[dict]):
    pilot.PILOT_DIR = run_dir
    out_dir = run_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    category_ids = settings.category_name_to_id
    images, annotations, rejected_all, queue = [], [], {}, []
    totals = Counter()

    state_path = run_dir / "state.json"
    damaged = set(json.loads(state_path.read_text()).get("damaged", [])) if state_path.exists() else set()
    for photo in photos:
        boxes, rejected, info = photo_boxes(photo, photo["id"] in damaged)
        counts = Counter()
        for b in boxes:
            source = SOURCES[b["source"]]
            if b["source"] == "existing":
                change = "box_corrected" if b["status"] == "fixed" else "unchanged"
            else:
                change = "added"
            attributes = {"source": source, "change": change}
            if b["confidence"]:
                attributes["confidence"] = b["confidence"]
            if b.get("original_box"):
                attributes["original_bbox"] = xywh(b["original_box"])
            bbox = xywh(b["box"])
            annotations.append({"id": len(annotations) + 1, "image_id": photo["id"],
                                "category_id": category_ids[b["class"]], "bbox": bbox,
                                "area": round(bbox[2] * bbox[3], 2), "iscrowd": 0, "attributes": attributes})
            counts[f"{source} {change}" + (f" {b['confidence']}" if b["confidence"] else "")] += 1
            counts[b["class"]] += 1
        uncertain = (sum(1 for b in boxes if b["confidence"] == "low")
                     + sum(1 for b in boxes if b["source"] == "review" and b["confidence"] == "medium")
                     + sum(1 for b in boxes if b["class"] == "non-fish" and b["confidence"] != "low")
                     + len(rejected))
        priority = uncertain + 3 * info["crowded"] + 5 * (info["unusable"] or info["status"] != "reviewed")
        image = {k: photo[k] for k in ("id", "file_name", "width", "height") if k in photo}
        image.update({k: photo[k] for k in ("datetime", "camera") if k in photo})
        image["vlm"] = {**info, "check_priority": priority, "counts": dict(counts), "rejected_candidates": len(rejected)}
        images.append(image)
        if rejected:
            rejected_all[photo["file_name"]] = [{"bbox": xywh(r["box"]), "class": r["class"], "source": r["source"],
                                                 "confidence": r["confidence"]} for r in rejected]
        queue.append((priority, photo["file_name"], photo["id"], info, counts, len(rejected)))
        totals.update(counts)
        totals[f"images {info['status']}"] += 1
        totals["images unusable"] += info["unusable"]
        totals["images crowded"] += info["crowded"]
        totals["rejected candidates"] += len(rejected)

    cost_path = run_dir / "cost.json"
    coco = {
        "info": {
            "description": f"{pilot.DATASET} with VLM pre-annotation for human review (tools/vlm_annotation)",
            "date_created": time.strftime("%Y-%m-%d"),
            "pass1": {"variant": PASS1, **pilot.VARIANTS[PASS1]},
            "pass2": {"reviewer": REVIEWER, **second_pass.REVIEWERS[REVIEWER]},
            "merge_policy": POLICY,
            "cost": json.loads(cost_path.read_text()) if cost_path.exists() else None,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": settings.coco_categories,
    }
    (out_dir / f"{pilot.DATASET}_vlm_coco.json").write_text(json.dumps(coco))
    (out_dir / "rejected_candidates.json").write_text(json.dumps(rejected_all, indent=1))
    with open(out_dir / "review_queue.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check_priority", "file_name", "image_id", "status", "unusable", "crowded", "human_unchanged",
                         "human_box_corrected", "added_high", "added_medium", "added_low", "rejected_candidates"])
        for priority, file_name, image_id, info, counts, n_rejected in sorted(queue, key=lambda q: (-q[0], q[1])):
            added = Counter()
            for key, n in counts.items():
                if key.startswith(("vlm_pass1 added", "vlm_review added")):
                    added[key.rsplit(" ", 1)[1]] += n
            writer.writerow([priority, file_name, image_id, info["status"], int(info["unusable"]), int(info["crowded"]),
                             counts["human unchanged"], counts["human box_corrected"], added["high"], added["medium"],
                             added["low"], n_rejected])
    summary = {"images": len(images), "annotations": len(annotations), **dict(sorted(totals.items())),
               "check_priority_0": sum(1 for q in queue if q[0] == 0),
               "check_priority_1_to_4": sum(1 for q in queue if 1 <= q[0] <= 4),
               "check_priority_5_plus": sum(1 for q in queue if q[0] >= 5)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"Export written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("data/vlm_annotation") / pilot.DATASET)
    args = parser.parse_args()
    import batch_run

    run = batch_run.Run(args.run_dir, None)
    export(run.dir, run.photos)


if __name__ == "__main__":
    sys.exit(main())
