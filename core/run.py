"""Workspace entry point; only the local source package is added to sys.path."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.cli import main
if __name__ == '__main__':
    main()
