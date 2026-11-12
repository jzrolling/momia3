# BaSiC illumination correction

momia3 vendors the lab PyTorch adaptation of BaSiCPy as
momia3.vendor.basicpy_torch and exposes it as
preprocess.illumination_basicpy.

## Reference

- Peng, T. et al. "BaSiC: a background and shading correction method for
  optical microscopy images." Nature Communications 8, 14836 (2017).
  DOI: 10.1038/s41467-017-01613-0
- Upstream implementation: https://github.com/peng-lab/BaSiCPy (MIT)
- Vendored source: src/momia3/vendor/basicpy_torch/; provenance and the
  list of changes are in its README.md. License text is in
  src/momia3/vendor/basicpy_torch/LICENSE.

## Usage

    from momia3.ops.basic_correction import BasicPyParams, basicpy_correct
    corrected = basicpy_correct(stack_t_y_x, BasicPyParams(device="cpu"))

basicpy_fit returns the fitted model for reuse with basicpy_transform.
The package requires torch and is imported lazily; importing momia3 or
momia3.ops.preprocess does not load torch or the vendored package.
