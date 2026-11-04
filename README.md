# momia3

Single-bacterium microscopy image characterization and lineage tracking platform.
Ground-up refactor of MOMIA2: Arrow/Zarr data layer, typed operator registry
(CLI / GUI / MCP schemas), sub-pixel outline and midline extraction, and optional
Omnipose inference.

## Install

    git clone git@github.com:jzrolling/momia3.git
    cd momia3
    pip install -e .
    pip install -e ".[inference]"    # optional: Omnipose-lite inference backend
    pip install -e ".[gui]"          # optional: napari plugin

Python >= 3.11. Core install needs no torch.

## Quick start

    momia3 run examples/pipeline_demo.json          # synthetic end-to-end demo
    momia3 run examples/pipeline_sp003_omnipose.json # real data + Omnipose (weights required)
    momia3 list                                     # registered operations

The pipeline DAG is plain JSON: preprocess -> segment -> measure -> outline ->
midline -> characterize -> track.

## Optional Omnipose inference

momia3.models.omnipose_lite runs the reference Omnipose CPnet weights with a
lightweight orchestration (no RGB flow rendering, torch Euler integration).
Weights are downloaded separately and never bundled:

    from momia3.models.omnipose_lite import FastOmnipose
    fast = FastOmnipose(device="mps")               # resolves ~/.cellpose/models/...
    masks = fast.segment(image)

Set MOMIA3_OMNIPOSE_WEIGHTS to use another checkpoint. See
docs/omnipose_lite.md.

## Documentation

- ARCHITECTURE.md - data model, operator spec, midline/outline design
- docs/outline_midline.md - outline/midline parameters, pipeline usage, benchmarks
- docs/omnipose_lite.md - inference backend, weights, preprocessing notes

## Status

Research software under active development (momia2 refactor, Phase 2/3).
APIs and on-disk schemas may change.
