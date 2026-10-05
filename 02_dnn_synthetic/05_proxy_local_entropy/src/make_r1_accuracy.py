#!/usr/bin/env python3
"""Rebuild r=1 accuracy from the original saved per-reference observations."""
from pathlib import Path
from utils.r1_accuracy import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
