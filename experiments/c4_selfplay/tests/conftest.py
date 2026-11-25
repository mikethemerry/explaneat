import sys
from pathlib import Path

# Study code is imported as top-level modules (engine, mcts, ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
