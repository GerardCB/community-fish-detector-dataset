"""
Input of the VLM pre-annotation for the KSO Mussel Farm BRUV dataset (Zenodo 14713926).

Reads the Roboflow YOLO export (data/raw/kso_mussel_bruv/model_dataset/{train,valid,test}) and
writes data/vlm_annotation/kso_mussel_bruv_input/: the 910 frames under their original names
(the Roboflow suffix "_jpg.rf.<hash>" removed) and a COCO file with the human boxes as fish /
non-fish. Gastropoda boxes (pale snails of 2-3 cm, size doubtful) are left out, and the prompt
tells the models not to box them.

Usage (from the repo root): venv-vlm/bin/python tools/vlm_annotation/prepare_kso_mussel_bruv.py
then: VLM_DATASET=kso_mussel_bruv venv-vlm/bin/python tools/vlm_annotation/batch_run.py run
"""

import json
import re
import shutil
from pathlib import Path

from PIL import Image

SOURCE = Path("data/raw/kso_mussel_bruv/model_dataset")
OUT = Path("data/vlm_annotation/kso_mussel_bruv_input")
SOURCE_CLASSES = ["Gastropoda sp", "Gobiidae sp", "Hyas araneus", "Paguridae sp", "Pleuronectiformes sp",
                  "Portunoidea sp"]
CLASS = {"Gobiidae sp": "fish", "Pleuronectiformes sp": "fish", "Hyas araneus": "non-fish",
         "Paguridae sp": "non-fish", "Portunoidea sp": "non-fish"}  # Gastropoda sp: left out
NAME = re.compile(r"^((?:control|farm)_\d+_\d+_frame_\d+)_jpg\.rf\.[0-9a-f]+\.jpg$")


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish"}, {"id": 2, "name": "non-fish"}]}
    left_out = 0
    for split in ("train", "valid", "test"):
        for path in sorted((SOURCE / split / "images").glob("*.jpg")):
            file_name = NAME.match(path.name)[1] + ".jpg"
            shutil.copy2(path, OUT / "images" / file_name)
            with Image.open(path) as image:
                width, height = image.size
            image_id = len(coco["images"]) + 1
            coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height,
                                   "roboflow_split": split})
            for line in (SOURCE / split / "labels" / (path.stem + ".txt")).read_text().splitlines():
                values = line.split()
                if not values:
                    continue
                name = SOURCE_CLASSES[int(values[0])]
                if name not in CLASS:
                    left_out += 1
                    continue
                coords = [float(v) for v in values[1:]]
                if len(coords) == 4:
                    cx, cy, w, h = coords
                    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                else:  # a polygon
                    xs, ys = coords[0::2], coords[1::2]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                x0, y0 = max(0.0, x0) * width, max(0.0, y0) * height
                x1, y1 = min(1.0, x1) * width, min(1.0, y1) * height
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                            "category_id": 1 if CLASS[name] == "fish" else 2,
                                            "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                            "iscrowd": 0, "source_class": name})
    (OUT / "annotations_coco.json").write_text(json.dumps(coco))
    print(f"{len(coco['images'])} frames, {len(coco['annotations'])} human boxes ({left_out} Gastropoda left out) "
          f"written to {OUT}")


if __name__ == "__main__":
    main()
