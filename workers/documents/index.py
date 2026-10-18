"""VISION runtime worker: documents (office text documents, LaTeX)."""
from runtime.worker_base import Worker

WORKER = Worker("documents")
run = WORKER.run
