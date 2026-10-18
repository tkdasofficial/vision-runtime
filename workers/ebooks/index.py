"""VISION runtime worker: ebooks. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("ebooks")
run = WORKER.run
