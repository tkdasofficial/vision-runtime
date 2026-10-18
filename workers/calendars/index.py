"""VISION runtime worker: calendars. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("calendars")
run = WORKER.run
