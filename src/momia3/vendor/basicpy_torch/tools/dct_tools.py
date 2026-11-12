"""2D and 3D discrete cosine transform tools.

This file contains PyTorch implementations of DCT/IDCT operations that replaced the original JAX implementations.
The conversion was assisted by doubao-seed-2.0-Code.

Key changes from JAX to PyTorch:
- Replaced JAX dct/idct functions with PyTorch-compatible implementations
- Added TorchDCT class to handle PyTorch tensors with SciPy backend
- Maintained the same DCT/IDCT interface for compatibility
- Added proper device handling for GPU support
"""

import importlib.util
import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from functools import partial

import numpy as np
import scipy.fft
import torch

__all__ = [
    "dct2d",
    "idct2d",
    "dct3d",
    "idct3d",
]

# initialize logger with the package name
logger = logging.getLogger(__name__)


class DCT(ABC):
    @abstractproperty
    @abstractmethod
    def _backend(self) -> str:
        ...

    @staticmethod
    @abstractmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        ...




class SciPyDCT(DCT):
    _backend = "SCIPY"

    @staticmethod
    def dctnd(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.dctn(arr, norm="ortho")

    @staticmethod
    def idctnd(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.idctn(arr, norm="ortho")

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.dctnd(arr)

    @staticmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.dctnd(arr)

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.idctnd(arr)

    @staticmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.idctnd(arr)


class TorchDCT(DCT):
    _backend = "PYTORCH"

    @staticmethod
    def dct2d(arr: torch.Tensor) -> torch.Tensor:
        """2D DCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply 2D DCT using SciPy
        dct_np = scipy.fft.dctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(dct_np, dtype=arr.dtype, device=arr.device)

    @staticmethod
    def idct2d(arr: torch.Tensor) -> torch.Tensor:
        """2D IDCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply 2D IDCT using SciPy
        idct_np = scipy.fft.idctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(idct_np, dtype=arr.dtype, device=arr.device)

    @staticmethod
    def dct3d(arr: torch.Tensor) -> torch.Tensor:
        """3D DCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply 3D DCT using SciPy
        dct_np = scipy.fft.dctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(dct_np, dtype=arr.dtype, device=arr.device)

    @staticmethod
    def idct3d(arr: torch.Tensor) -> torch.Tensor:
        """3D IDCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply 3D IDCT using SciPy
        idct_np = scipy.fft.idctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(idct_np, dtype=arr.dtype, device=arr.device)

    @staticmethod
    def dctnd(arr: torch.Tensor) -> torch.Tensor:
        """ND DCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply ND DCT using SciPy
        dct_np = scipy.fft.dctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(dct_np, dtype=arr.dtype, device=arr.device)

    @staticmethod
    def idctnd(arr: torch.Tensor) -> torch.Tensor:
        """ND IDCT implementation for PyTorch using SciPy."""
        # Convert to numpy array
        arr_np = arr.cpu().numpy()
        # Apply ND IDCT using SciPy
        idct_np = scipy.fft.idctn(arr_np, norm="ortho")
        # Convert back to PyTorch tensor
        return torch.tensor(idct_np, dtype=arr.dtype, device=arr.device)


# collect all subclasses into a dictionary
DCT_BACKENDS = {sc()._backend: sc() for sc in DCT.__subclasses__()}  # type: ignore
