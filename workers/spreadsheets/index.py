"""VISION runtime worker: spreadsheets. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("spreadsheets")
run = WORKER.run
