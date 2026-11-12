# Camera corrections: dark/bias, hot/dead pixels, vignetting

These operators are opt-in preprocessing steps; the default pipeline is
unchanged.

- preprocess.dark_correct: subtract a dark frame or a per-frame border/overscan
  bias estimate. Inputs: image plus optional dark frame/stack.
- preprocess.hot_pixel_correct: replace hot/dead pixels by a local median.
  Threshold is either robust sigma (default) or absolute camera units.
- preprocess.vignetting_correct: divide by a smooth unit-mean gain surface,
  either supplied by the caller or estimated from a stack percentile.

References:

- BaSiC: Peng, T. et al. Nature Communications 8, 14836 (2017),
  DOI 10.1038/s41467-017-01613-0.
- CIDRE: Smith, K. et al. Nature Methods 12, 404-406 (2015).
- Dark-frame and flat-field calibration are standard camera calibration
  practice; hot/dead-pixel replacement uses robust median residuals.
