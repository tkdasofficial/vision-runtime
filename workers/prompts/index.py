"""Compatibility entry point; story and prompt generation share one extracted pipeline."""
import sys
from workers.prompts.story import index as implementation
sys.modules[__name__] = implementation
