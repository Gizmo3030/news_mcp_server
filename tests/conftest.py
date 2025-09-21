import os
import sys
import pathlib

# Ensure src is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Default to Ollama in tests; we stub network anyway
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
