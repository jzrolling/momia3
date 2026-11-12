"""Types for PyTorch implementation.

This file contains type definitions that have been updated to support PyTorch tensors,
replacing the original JAX array types. The conversion was assisted by doubao-seed-2.0-Code.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]  # dask.array.Array, zarr.Array
PathLike = Union[str, Path]
