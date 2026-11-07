#!/usr/bin/env python3
"""Analyse an image and print its category + suggested parameters as JSON.

usage: python3 analyze.py IMAGE
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2v.imgio import load_rgba  # noqa: E402
from p2v.analyze import analyze  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    rgba, has_alpha = load_rgba(sys.argv[1])
    print(json.dumps(analyze(rgba, has_alpha), indent=2))


if __name__ == "__main__":
    main()
