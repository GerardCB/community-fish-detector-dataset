"""
Review sheets of a pre-annotation run (no API calls), to check the boxes by eye.

  frames   before (human boxes) and after (export) for photos listed in a TSV with a file_name column,
           or for --n random photos; added boxes coloured by confidence
  crops    enlarged crops of added boxes, 48 per sheet, each labelled with its key (a<annotation id>,
           as used in the decisions file), class, source and confidence: --confidence high|medium|low
           selects them, --n samples them at random; --added <file> takes the boxes of a selection
           (verify_added.py select) instead, e.g. --added datasets/<name>_added_boxes.json --n 100 for the
           precision check
  static   one row per static group (verify_added.py static): up to 8 crops from different frames,
           labelled with the group index, for the object_groups / animal_groups of the decisions file

Usage (from the repo root): VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/sheets.py crops \
    --confidence high [--run-dir DIR] [--n N]
Sheets are written to <run dir>/sheets/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pilot  # noqa: E402

COLORS = {"human": (0, 220, 0), "high": (255, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 255)}
TILE = 220


def load_export(run_dir: Path) -> tuple[dict, dict, dict]:
    """(images by id, annotations by image id, image id by file name) of the run's export."""
    coco = json.loads((run_dir / "export" / f"{pilot.DATASET}_vlm_coco.json").read_text())
    images = {image["id"]: image for image in coco["images"]}
    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    return images, by_image, {image["file_name"]: image["id"] for image in coco["images"]}


def read(file_name: str):
    return cv2.imread(str(pilot.IMAGES_DIR / file_name))


def crop_tile(photo, bbox, label: str, color):
    """An enlarged crop around an [x, y, w, h] box, with the box drawn and a label below."""
    height, width = photo.shape[:2]
    x, y, w, h = bbox
    pad = max(20, 0.8 * max(w, h))
    x0, y0, x1, y1 = int(max(0, x - pad)), int(max(0, y - pad)), int(min(width, x + w + pad)), int(min(height, y + h + pad))
    crop = photo[y0:y1, x0:x1]
    s = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (int((x - x0) * s), int((y - y0) * s)), (int((x + w - x0) * s), int((y + h - y0) * s)), color, 2)
    tile = np.full((TILE + 22, TILE, 3), 30, np.uint8)
    tile[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(tile, label[:34], (3, TILE + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return tile


def write_grid(tiles: list, columns: int, path: Path, title: str):
    blank = np.full_like(tiles[0], 30)
    tiles = tiles + [blank] * (-len(tiles) % columns)
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    head = np.full((30, sheet.shape[1], 3), 30, np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def frames(args, out: Path):
    images, by_image, ids = load_export(args.run_dir)
    if args.images:
        lines = args.images.read_text().splitlines()
        column = lines[0].split("\t").index("file_name")
        names = [line.split("\t")[column] for line in lines[1:] if line.strip()]
    else:
        names = random.Random(args.seed).sample(sorted(ids), min(args.n, len(ids)))
    for k, name in enumerate(names, 1):
        photo = read(name)
        before, after = photo.copy(), photo.copy()
        for a in by_image.get(ids[name], []):
            x, y, w, h = (int(v) for v in a["bbox"])
            human = a["attributes"]["source"] == "human"
            color = COLORS["human"] if human else COLORS[a["attributes"].get("confidence", "low")]
            if human and a["attributes"]["change"] == "unchanged":
                cv2.rectangle(before, (x, y), (x + w, y + h), color, 2)
            elif human:
                ox, oy, ow, oh = (int(v) for v in a["attributes"]["original_bbox"])
                cv2.rectangle(before, (ox, oy), (ox + ow, oy + oh), color, 2)
                color = (255, 128, 0)  # tightened human box
            cv2.rectangle(after, (x, y), (x + w, y + h), color, 2)
            if a["category_id"] == 2:
                cv2.putText(after, "NF", (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        row = np.hstack([before, after])
        scale = min(1.0, 1600 / row.shape[1])
        row = cv2.resize(row, None, fx=scale, fy=scale) if scale < 1 else row
        head = np.full((30, row.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{name}  left: human boxes; right: after (green human, blue tightened, magenta/orange/"
                    "yellow added high/medium/low, NF non-fish)", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"frame_{k:03d}.jpg"), np.vstack([head, row]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(names)} frame sheets written to {out}")


def crops(args, out: Path):
    if args.added:
        # a selection: {file name without the prefix: [{bbox, class, source, confidence, verified}]}
        selection = json.loads(args.added.read_text())
        boxes = [({"bbox": e["bbox"], "category_id": 2 if e["class"] == "non-fish" else 1, "id": f"{k}#{i}",
                   "attributes": {**e, "source": e["source"]}}, pilot.DATASET_CONFIG["prefix"] + k)
                 for k, entries in sorted(selection.items()) for i, e in enumerate(entries)]
    else:
        images, by_image, _ = load_export(args.run_dir)
        boxes = [(a, images[a["image_id"]]["file_name"]) for anns in by_image.values() for a in anns
                 if a["attributes"]["source"] != "human"]
        boxes.sort(key=lambda b: b[0]["id"])
    if args.confidence:
        boxes = [b for b in boxes if b[0]["attributes"].get("confidence") == args.confidence]
    if args.n:
        boxes = random.Random(args.seed).sample(boxes, min(args.n, len(boxes)))
    tiles = []
    for k, (a, name) in enumerate(boxes, 1):
        attributes = a["attributes"]
        key = f"#{k}" if args.added else f"a{a['id']}"
        label = (f"{key} {'NF' if a['category_id'] == 2 else 'fish'} {attributes['source'].removeprefix('vlm_')} "
                 f"{attributes.get('confidence', '')}")
        tiles.append(crop_tile(read(name), a["bbox"], label, COLORS[attributes.get("confidence", "low")]))
    what = ("selection" if args.added else "") + (args.confidence or ("" if args.added else "all"))
    for start in range(0, len(tiles), 48):
        write_grid(tiles[start:start + 48], 8, out / f"crops_{what}_{start // 48 + 1:02d}.jpg",
                   f"added boxes ({what}), {start + 1}-{min(len(tiles), start + 48)} of {len(tiles)}")
    print(f"{len(tiles)} crops in {-(-len(tiles) // 48)} sheets written to {out}")


def static(args, out: Path):
    groups = json.loads((args.run_dir / "verify" / "static_groups.json").read_text())
    rows = []
    for index, group in enumerate(groups):
        seen, tiles = set(), []
        for b in group:
            if b["file_name"] in seen or len(tiles) == 8:
                continue
            seen.add(b["file_name"])
            tiles.append(crop_tile(read(b["file_name"]), b["bbox"], f"#{index} {b['class']} {b['confidence']}",
                                   COLORS[b["confidence"]]))
        tiles += [np.full_like(tiles[0], 30)] * (8 - len(tiles))
        rows.append(np.hstack(tiles))
    for start in range(0, len(rows), 6):
        head = np.full((30, rows[0].shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"static groups {start}-{min(len(rows), start + 6) - 1} of {len(rows)}: is each group an "
                    "animal (keep) or an object (object_groups)?", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"static_{start // 6 + 1:02d}.jpg"), np.vstack([head] + rows[start:start + 6]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(rows)} static groups written to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["frames", "crops", "static"])
    parser.add_argument("--run-dir", type=Path, default=Path("data/vlm_annotation") / pilot.DATASET)
    parser.add_argument("--images", type=Path, help="frames: TSV with a file_name column")
    parser.add_argument("--confidence", choices=["high", "medium", "low"], help="crops: only this confidence")
    parser.add_argument("--added", type=Path, help="crops: boxes of this selection file instead of the export")
    parser.add_argument("--n", type=int, default=0, help="frames / crops: random sample size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "frames" and not args.images and not args.n:
        args.n = 12
    out = args.run_dir / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    {"frames": frames, "crops": crops, "static": static}[args.command](args, out)


if __name__ == "__main__":
    sys.exit(main())
