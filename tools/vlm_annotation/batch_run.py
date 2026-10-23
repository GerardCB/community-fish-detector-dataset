"""
Two-pass VLM pre-annotation of a dataset with the Message Batches API (half price). The dataset is the
config named by VLM_DATASET (see pilot.py).

Pass 1 (pilot.py variant claude-haiku-5-5_tiled_v2) proposes fish / non-fish boxes in 4 overlapping
tiles of each photo. Pass 2 (second_pass.py reviewer review_haiku_v2) reviews, on each photo, the
human boxes and the boxes pass 1 added: it tightens loose boxes, flags wrong ones and adds missed
animals. `export` merges both into a COCO file for the human annotators (export.py). See README.md.

Every stage is resumable. The submitted batches are in <run dir>/state.json and every result is
appended to <run dir>/results/<stage>.jsonl; a stage only submits requests that have no usable
result and are not already in a submitted batch. Failed requests are resubmitted, up to
MAX_ATTEMPTS attempts each. Requests carry their image inline, so they are split into batches of
at most MAX_BATCH_BYTES.

Usage (from the repo root, with ANTHROPIC_API_KEY in .env):
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py run       # pass 1, pass 2 and the export
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py status
  VLM_DATASET=<name> venv-vlm/bin/python tools/vlm_annotation/batch_run.py cost
Options: --run-dir (default data/vlm_annotation/<name>), --images <tsv with a file_name
column> to process only those photos. `run` waits for each batch (up to 24 hours each); start it in
tmux or nohup.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import pilot  # noqa: E402
import second_pass  # noqa: E402

PASS1 = second_pass.FIRST_PASS
REVIEWER = "review_haiku_v2"
STAGES = {"pass1": "p1", "pass2": "p2"}
MAX_BATCH_BYTES = 200_000_000  # the API limit is 256 MB per batch
MAX_BATCH_REQUESTS = 50_000  # the API limit is 100,000
MAX_ATTEMPTS = 3
POLL_SECONDS = 60
BATCH_PRICE_FACTOR = 0.5
DEFAULT_RUN_DIR = Path("data/vlm_annotation") / pilot.DATASET
RAW_IMAGES_DIR = Path(pilot.DATASET_CONFIG["raw_images_dir"] or pilot.DATASET_CONFIG["images_dir"])


def log(message: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def with_retries(what: str, fn, attempts: int = 30):
    """Call fn(), waiting and retrying on rate limits, connection problems and server errors."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            wait = min(600, 30 * attempt)
            log(f"{what}: {type(e).__name__} ({e}); retrying in {wait} s ({attempt}/{attempts})")
            time.sleep(wait)
    return fn()


class Run:
    def __init__(self, run_dir: Path, image_list: Path | None):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "results").mkdir(exist_ok=True)
        pilot.PILOT_DIR = run_dir  # per-photo records go to <run dir>/raw/
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "images": str(image_list) if image_list else "all", "stages": {s: {"batches": []} for s in STAGES}}
        wanted = self.state["images"]
        if image_list and wanted != str(image_list):
            raise SystemExit(f"{run_dir} was started with --images {wanted}; use another --run-dir")
        photos = self.frozen_inputs()
        if wanted != "all":
            lines = Path(wanted).read_text().splitlines()
            column = lines[0].split("\t").index("file_name")
            names = {line.split("\t")[column] for line in lines[1:] if line.strip()}
            photos = [p for p in photos if p["file_name"] in names]
        self.photos = photos
        self.photo_by_id = {p["id"]: p for p in photos}
        if "damaged" not in self.state:
            self.state["damaged"] = self.find_damaged()
            self.save()
        self.damaged = set(self.state["damaged"])

    def frozen_inputs(self) -> list[dict]:
        """
        The photo list (with ids) and the human boxes of the run, copied into the run dir on its
        first start, so that rebuilding the dataset's data/processing folder during a run (which can
        renumber the photos) changes nothing. Photos are read from raw_images_dir when present.
        """
        photos_path, boxes_path = self.dir / "photos.json", self.dir / "human_boxes.json"
        classes_path = self.dir / "human_classes.json"
        if not photos_path.exists() or not boxes_path.exists():
            coco = json.loads(pilot.ANNOTATIONS_PATH.read_text())
            photos_path.write_text(json.dumps(sorted(coco["images"], key=lambda i: i["file_name"])))
            boxes_path.write_text(json.dumps(pilot.existing_boxes_by_file()))
            if pilot.DATASET_CONFIG["human_classes"]:
                classes_path.write_text(json.dumps(pilot.existing_classes_by_file()))
        human_boxes = json.loads(boxes_path.read_text())
        pilot.existing_boxes_by_file = lambda: human_boxes
        if classes_path.exists():
            human_classes = json.loads(classes_path.read_text())
            pilot.existing_classes_by_file = lambda: human_classes
        return json.loads(photos_path.read_text())

    @staticmethod
    def image_path(file_name: str) -> Path:
        raw = RAW_IMAGES_DIR / file_name.removeprefix(pilot.DATASET_CONFIG["prefix"])
        return raw if raw.exists() else pilot.IMAGES_DIR / file_name

    def find_damaged(self) -> list[int]:
        """Photos whose file cannot be decoded completely (e.g. 2 truncated files in the OBSEA download)."""
        damaged = []
        for photo in self.photos:
            try:
                with Image.open(self.image_path(photo["file_name"])) as image:
                    image.load()
            except OSError as e:
                log(f"damaged photo, skipped: {photo['file_name']} ({e})")
                damaged.append(photo["id"])
        return damaged

    def save(self):
        partial = self.state_path.with_name("state.json.part")
        partial.write_text(json.dumps(self.state))
        partial.rename(self.state_path)

    # Results -----------------------------------------------------------------------------------

    def results_path(self, stage: str) -> Path:
        return self.dir / "results" / f"{stage}.jsonl"

    def results(self, stage: str) -> dict[tuple[int, int], list[dict]]:
        """{(image id, job index): result rows in submission order}."""
        rows = defaultdict(list)
        path = self.results_path(stage)
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                rows[(row["image_id"], row["job"])].append(row)
        for attempts in rows.values():
            attempts.sort(key=lambda r: r["attempt"])
        return rows

    @staticmethod
    def parsed(stage: str, row: dict):
        """The answer of a result row, or None if it failed."""
        call = row.get("call") or {}
        if row["type"] != "succeeded" or "error" in call or not call.get("text"):
            return None
        answer = json.loads(call["text"])
        return answer[pilot.pass1_key(PASS1)] if stage == "pass1" else answer

    # Jobs ----------------------------------------------------------------------------------------

    def jobs(self, stage: str, photo: dict) -> tuple[list[dict], list[dict] | None]:
        """The requests of a photo for a stage (and, for pass 2, the boxes it reviews)."""
        if stage == "pass1":
            return pilot.pass1_jobs(PASS1, photo["width"], photo["height"]), None
        boxes = second_pass.input_boxes(photo["file_name"])
        return second_pass.review_jobs(REVIEWER, photo["width"], photo["height"], boxes), boxes

    def stage_photos(self, stage: str) -> list[dict]:
        if stage == "pass1":
            return [p for p in self.photos if p["id"] not in self.damaged]
        # Pass 2 reviews the photos whose pass-1 record is complete
        ok = []
        for photo in self.photos:
            path = pilot.raw_path(PASS1, photo["file_name"])
            if path.exists() and "error" not in json.loads(path.read_text()):
                ok.append(photo)
        return ok

    def in_flight(self, stage: str) -> set[tuple[int, int]]:
        keys = set()
        for batch in self.state["stages"][stage]["batches"]:
            if not batch.get("collected"):
                keys.update((int(i), int(j)) for i, j, _ in (c.split("-")[1:] for c in batch["custom_ids"]))
        return keys

    def pending(self, stage: str) -> list[tuple[dict, int, int]]:
        """(photo, job index, attempt) of every request to submit now."""
        results, flying, todo = self.results(stage), self.in_flight(stage), []
        for photo in self.stage_photos(stage):
            jobs, _ = self.jobs(stage, photo)
            for j, job in enumerate(jobs):
                key = (photo["id"], j)
                if key in flying:
                    continue
                attempts = results.get(key, [])
                answers = [a for a in (self.parsed(stage, r) for r in attempts) if a is not None]
                if len(attempts) >= MAX_ATTEMPTS:
                    continue
                if not answers:
                    todo.append((photo, j, len(attempts)))
                elif stage == "pass2" and len(answers) == 1 and second_pass.needs_retry(job, answers[0]):
                    todo.append((photo, j, len(attempts)))  # asked once more, as in the pilot
        return todo

    # Submission and collection ---------------------------------------------------------------

    def submit(self, client: anthropic.Anthropic, stage: str, todo: list[tuple[dict, int, int]]):
        chunk, size = [], 0

        def flush():
            nonlocal chunk, size
            if not chunk:
                return
            batch = with_retries("create batch", lambda: client.with_options(timeout=1800, max_retries=2)
                                 .messages.batches.create(requests=chunk))
            self.state["stages"][stage]["batches"].append(
                {"id": batch.id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes": size,
                 "custom_ids": [r["custom_id"] for r in chunk], "collected": False})
            self.save()
            log(f"{stage}: submitted batch {batch.id} with {len(chunk)} requests ({size / 1e6:.0f} MB)")
            chunk, size = [], 0

        by_photo = defaultdict(list)
        for photo, j, attempt in todo:
            by_photo[photo["id"]].append((j, attempt))
        for image_id, items in by_photo.items():
            photo = self.photo_by_id[image_id]
            jobs, boxes = self.jobs(stage, photo)
            with Image.open(self.image_path(photo["file_name"])) as image:
                image = image.convert("RGB")
                for j, attempt in items:
                    if stage == "pass1":
                        params = pilot.pass1_request(PASS1, image, jobs[j])
                    else:
                        params = second_pass.review_request(REVIEWER, image, boxes, jobs[j])
                    request = {"custom_id": f"{STAGES[stage]}-{image_id}-{j}-{attempt}", "params": params}
                    request_size = len(json.dumps(request))
                    if chunk and (size + request_size > MAX_BATCH_BYTES or len(chunk) >= MAX_BATCH_REQUESTS):
                        flush()
                    chunk.append(request)
                    size += request_size
        flush()

    def wait_and_collect(self, client: anthropic.Anthropic, stage: str):
        key = pilot.pass1_key(PASS1) if stage == "pass1" else None
        while True:
            waiting = [b for b in self.state["stages"][stage]["batches"] if not b.get("collected")]
            if not waiting:
                return
            for batch in waiting:
                info = with_retries("retrieve batch", lambda: client.messages.batches.retrieve(batch["id"]))
                if info.processing_status != "ended":
                    continue

                def download():
                    rows = []
                    for result in client.messages.batches.results(batch["id"]):
                        _, image_id, j, attempt = result.custom_id.split("-")
                        row = {"custom_id": result.custom_id, "batch_id": batch["id"], "image_id": int(image_id),
                               "job": int(j), "attempt": int(attempt), "type": result.result.type}
                        if result.result.type == "succeeded":
                            row["call"], _ = pilot.parse_message(result.result.message, key)
                        elif result.result.type == "errored":
                            error = getattr(result.result.error, "error", result.result.error)
                            row["error"] = f"{getattr(error, 'type', 'error')}: {getattr(error, 'message', error)}"
                        rows.append(row)
                    return rows

                rows = with_retries("download results", download)  # all rows first, so a broken download writes nothing
                with open(self.results_path(stage), "a") as out:
                    out.writelines(json.dumps(row) + "\n" for row in rows)
                batch["collected"] = True
                batch["counts"] = info.request_counts.to_dict()
                self.save()
                log(f"{stage}: collected batch {batch['id']}: {batch['counts']}")
            if any(not b.get("collected") for b in self.state["stages"][stage]["batches"]):
                counts = [with_retries("retrieve batch", lambda b=b: client.messages.batches.retrieve(b["id"])).request_counts
                          for b in self.state["stages"][stage]["batches"] if not b.get("collected")]
                log(f"{stage}: waiting for {len(counts)} batches "
                    f"({sum(c.processing for c in counts)} requests processing)")
                time.sleep(POLL_SECONDS)

    def run_stage(self, client: anthropic.Anthropic, stage: str):
        while True:
            self.wait_and_collect(client, stage)
            todo = self.pending(stage)
            if not todo:
                break
            log(f"{stage}: {len(todo)} requests to submit")
            self.submit(client, stage, todo)
        self.write_records(stage)

    # Per-photo records --------------------------------------------------------------------------

    def write_records(self, stage: str):
        """One record per photo, in the format of pilot.py / second_pass.py, from the results."""
        results, n_ok, n_failed = self.results(stage), 0, 0
        for photo in self.stage_photos(stage):
            jobs, boxes = self.jobs(stage, photo)
            if stage == "pass1":
                outcomes = []
                for j in range(len(jobs)):
                    attempts = results.get((photo["id"], j), [])
                    good = [r for r in attempts if self.parsed(stage, r) is not None]
                    row = (good or attempts or [{"type": "missing", "error": "no result"}])[-1]
                    call = dict(row.get("call") or {"error": row.get("error", row["type"])})
                    outcomes.append((call, self.parsed(stage, row) if row.get("call") else None))
                record = pilot.pass1_merge(PASS1, photo["file_name"], photo["width"], photo["height"], jobs, outcomes)
                path = pilot.raw_path(PASS1, photo["file_name"])
            else:
                outcomes = []
                for j in range(len(jobs)):
                    attempts = results.get((photo["id"], j), [])
                    outcomes.append([(dict(r.get("call") or {"error": r.get("error", r["type"])}), self.parsed(stage, r))
                                     for r in attempts] or [({"error": "no result"}, None)])
                record = second_pass.review_merge(REVIEWER, photo["file_name"], photo["width"], photo["height"],
                                                  boxes, jobs, outcomes)
                path = second_pass.raw_path(REVIEWER, photo["file_name"])
            record["image_id"] = photo["id"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
            n_ok += "error" not in record
            n_failed += "error" in record
        log(f"{stage}: wrote {n_ok} photo records, {n_failed} with errors")

    # Cost ------------------------------------------------------------------------------------------

    def cost(self) -> dict:
        report = {"price_factor": BATCH_PRICE_FACTOR, "photos": len(self.photos), "stages": {}}
        for stage, model in (("pass1", pilot.VARIANTS[PASS1]["model"]), ("pass2", second_pass.REVIEWERS[REVIEWER]["model"])):
            tokens = {"input_tokens": 0, "output_tokens": 0}
            rows = [r for attempts in self.results(stage).values() for r in attempts]
            for r in rows:
                for k in tokens:
                    tokens[k] += (r.get("call") or {}).get("usage", {}).get(k, 0)
            price_in, price_out = pilot.PRICES[model]
            usd = (tokens["input_tokens"] * price_in + tokens["output_tokens"] * price_out) / 1e6 * BATCH_PRICE_FACTOR
            report["stages"][stage] = {"model": model, "requests": len(rows),
                                       "failed": sum(self.parsed(stage, r) is None for r in rows),
                                       **tokens, "usd": round(usd, 4)}
        report["usd_total"] = round(sum(s["usd"] for s in report["stages"].values()), 4)
        report["usd_per_photo"] = round(report["usd_total"] / max(1, len(self.photos)), 6)
        (self.dir / "cost.json").write_text(json.dumps(report, indent=1))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status", "cost"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--images", type=Path, help="TSV with a file_name column (default: all photos)")
    parser.add_argument("--stage", choices=["pass1", "pass2", "all"], default="all")
    args = parser.parse_args()
    pilot.load_env_file()
    run = Run(args.run_dir, args.images)
    client = anthropic.Anthropic()

    if args.command == "status":
        for stage in STAGES:
            batches = run.state["stages"][stage]["batches"]
            log(f"{stage}: {len(batches)} batches, {sum(not b.get('collected') for b in batches)} not collected")
            for b in batches:
                if not b.get("collected"):
                    info = client.messages.batches.retrieve(b["id"])
                    log(f"  {b['id']}: {info.processing_status} {info.request_counts.to_dict()}")
        return
    if args.command == "cost":
        print(json.dumps(run.cost(), indent=1))
        return

    log(f"{len(run.photos)} photos, run dir {run.dir}")
    for stage in (["pass1", "pass2"] if args.stage == "all" else [args.stage]):
        run.run_stage(client, stage)
    report = run.cost()
    log(f"cost: ${report['usd_total']:.2f} (${report['usd_per_photo']:.5f} per photo)")
    if args.stage in ("all", "pass2"):
        import export

        export.export(run.dir, run.photos)


if __name__ == "__main__":
    sys.exit(main())
