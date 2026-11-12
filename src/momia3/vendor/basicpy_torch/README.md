# BaSiCpy_torch (vendored)

This directory contains a complete, source-vendored copy of the lab PyTorch
port of BaSiCPy. It is a resident package inside momia3, not a pip dependency.

## Provenance

- Algorithm: BaSiC background and shading correction
  - Peng, T. et al. "BaSiC: a background and shading correction method for
    optical microscopy images." Nature Communications 8, 14836 (2017).
    DOI: 10.1038/s41467-017-01613-0
- Upstream implementation: BaSiCPy (JAX), MIT license
  - https://github.com/peng-lab/BaSiCPy
- PyTorch adaptation used as the direct source:
  - /Users/jzrolling/Desktop/Projects/Zlab_GUI/momia2-gui/external/BaSiCpy_torch
  - original conversion assisted by doubao-seed-2.0-Code; maintained by the
    lab as BaSiCpy_torch.
- License: see LICENSE (BaSiCPy, MIT, Copyright 2022 BaSiCPy developing
  collaboration group).

## Changes made for momia3

- Rewrote absolute BaSiCpy_torch imports to package-relative imports so the
  code lives under momia3.vendor.basicpy_torch.
- Fixed a scalar/tensor mismatch in _torch_routines.py: torch.sqrt(eta_D)
  with a Python integer is now an explicit float32 tensor on the data device.
- Removed caches, .DS_Store and Jupyter checkpoint files from the vendored copy.
- Import is intentionally lazy: momia3 core and momia3.ops.preprocess do not
  import this package; basicpy_torch requires torch.

The package is kept complete (BaSiC class, torch routines, DCT tools,
datasets/metrics helpers, CLI) so behavior stays close to the lab source.
