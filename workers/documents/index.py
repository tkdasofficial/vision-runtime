"""VISION runtime worker: documents (office text documents, LaTeX, and story generation in story.py)."""
from runtime.worker_base import Worker

WORKER = Worker("documents")
run = WORKER.run
