#!/bin/bash
cd "/Users/jzrolling/Desktop/Projects/Playground_kimi/momia3"
if [ "$1" = "--sp003" ]; then
  exec .venv/bin/python examples/launch_napari_sp003.py
else
  exec .venv/bin/python examples/launch_napari.py
fi
