"""Atomic per-cell benchmark progress, bound to the exact code, inputs and recipe.

The journal is authoritative; a partial CSV is useful output, never a completion
receipt. Failed cells are retried. A different recipe cannot reuse old scores.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd
from filelock import FileLock

from src.utils.atomic_output import replace_complete, write_csv
from src.utils.logging_setup import get_logger

DEADLINE_ENV = "CREDITICL_EVAL_DEADLINE"
PROGRESS_ENV = "CREDITICL_EVAL_PROGRESS"


class EvaluationDeferred(RuntimeError):
    """Allocation is ending; completed cells are durable and can be resumed."""


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            # Retain row/column order: reopening a completed journal must not
            # change its CSV bytes and invalidate an existing benchmark receipt.
            json.dump(data, stream, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_complete(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def identity(cfg, domain):
    from src.eval.baselines import TABICL_RELEASED, find_local_tabpfn_checkpoint
    from src.eval.benchmark_status import input_digest
    from src.utils import paths
    from src.utils.environment import environment_fingerprint
    from src.utils.preflight import source_fingerprint

    files = []
    if domain == "credit":
        from src.data.discovery import list_datasets
        from src.data.pipeline import ensure_processed

        names = cfg.datasets if cfg.datasets is not None else list_datasets(cfg.task)
        ensure_processed(cfg.task, names)
        for name in names:
            folder = paths.find_processed_dir(cfg.task, name)
            if folder is None:
                raise FileNotFoundError(f"Missing processed dataset {cfg.task}/{name}")
            files.extend(p for p in folder.iterdir() if p.is_file())
        tasks = [cfg.task]
    else:
        from src.eval.ood import list_ood_datasets, manifest_path, cache_path

        entries = [e for e in list_ood_datasets() if e.kind in cfg.kinds
                   and (cfg.datasets is None or e.name in cfg.datasets)]
        files.extend([manifest_path(), *(cache_path(e) for e in entries)])
        tasks = ["pd" if k == "classification" else "lgd" for k in cfg.kinds]
    for model in cfg.models:
        kw = cfg.model_kwargs.get(model, {})
        explicit = kw.get("checkpoint") or kw.get("model_path")
        if explicit:
            files.append(Path(explicit))
        elif model == "tabiclv2":
            files.extend(paths.find_pretrained(f"checkpoints/{TABICL_RELEASED[t]}") for t in tasks)
        elif model == "tabpfn3":
            for task in tasks:
                checkpoint = find_local_tabpfn_checkpoint("classifier" if task == "pd" else "regressor")
                if checkpoint is None:
                    raise FileNotFoundError(f"Missing released TabPFN {task} checkpoint")
                files.append(checkpoint)
        elif model == "crediticl":
            raise ValueError("Checkpoint required for resumable crediticl evaluation")
    recipe = dict(domain=domain, config=asdict(cfg), code=source_fingerprint(),
                  environment=environment_fingerprint(),
                  inputs={str(p.resolve()): input_digest(p) for p in sorted(set(files))})
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()


class CellJournal:
    def __init__(self, output, signature):
        self.output = Path(output)
        self.path = self.output.with_suffix(".progress.json")
        self.signature = signature
        self.rows = {}
        self.lock = FileLock(str(self.path) + ".lock", timeout=0)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if saved.get("schema") != 1 or saved.get("identity") != self.signature:
                    raise ValueError(f"Benchmark inputs/code/settings changed: {self.path}; use a fresh output namespace")
                self.rows = saved["cells"]
                if not isinstance(self.rows, dict) or any(self.key(r) != k for k, r in self.rows.items()):
                    raise ValueError(f"Malformed benchmark progress: {self.path}")
            elif self.output.exists():
                raise ValueError(f"Legacy results have no resumable provenance: {self.output}; use a fresh namespace")
            self.publish()
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *_):
        self.lock.release()

    @staticmethod
    def key(row):
        return json.dumps([row["dataset"], row["model"], int(row["seed"]), int(row["fold"])])

    def get(self, row):
        previous = self.rows.get(self.key(row))
        if previous and previous.get("status") == "ok":
            return dict(previous)
        return None

    def begin(self, row):
        deadline = os.environ.get(DEADLINE_ENV)
        estimates = [float(r.get("total_seconds", 0)) for r in self.rows.values()
                     if r.get("status") == "ok" and r["dataset"] == row["dataset"]
                     and r["model"] == row["model"]]
        reserve = max([120, *(t * 1.25 for t in estimates)])
        if deadline and time.time() + reserve >= float(deadline):
            raise EvaluationDeferred("Insufficient allocation time for the next complete fold; resubmit to continue")
        if os.environ.get(PROGRESS_ENV):
            atomic_json(os.environ[PROGRESS_ENV], dict(cell=self.key(row), started=time.time(),
                                                      completed=len([r for r in self.rows.values() if r["status"] == "ok"])))
        get_logger().info("[eval] START cell %s; saved successful cells=%d", self.key(row),
                          sum(r["status"] == "ok" for r in self.rows.values()))

    def save(self, row):
        self.rows[self.key(row)] = dict(row)
        self.publish()
        if os.environ.get(PROGRESS_ENV):
            atomic_json(os.environ[PROGRESS_ENV], dict(cell=None, started=None,
                                                      completed=sum(r["status"] == "ok" for r in self.rows.values())))

    def publish(self):
        atomic_json(self.path, dict(schema=1, identity=self.signature, cells=self.rows))
        # If interrupted between the journal and CSV writes, reopening recreates
        # the CSV from the already committed journal without recomputing a fold.
        if self.rows:
            write_csv(pd.DataFrame(self.rows.values()), self.output)
