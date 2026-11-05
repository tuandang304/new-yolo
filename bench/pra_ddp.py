"""Import target for Ultralytics DDP subprocesses: rebuilds the proposed-architecture trainer from
PRA_ARCH_JSON (set by bench.pra.make_trainer in the parent process)."""
import json
import os

from bench.pra import make_trainer

PRATrainer = make_trainer(json.loads(os.environ["PRA_ARCH_JSON"]))
