#!/bin/bash
cd "/Users/jzrolling/Desktop/Projects/Playground_kimi/momia3"
export QT_MAC_WANTS_LAYER=1
# Clear possibly-corrupt window/multi-monitor geometry that triggers
# degenerate GL viewports on macOS + PyQt6 (napari #6550 class of bugs).
.venv/bin/napari --reset >/dev/null 2>&1 || true
if [ "$1" = "--sp003" ]; then
  exec .venv/bin/python examples/launch_napari_sp003.py
else
  exec .venv/bin/python examples/launch_napari.py
fi
