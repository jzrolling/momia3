"""Main BaSiC class - PyTorch implementation derived from BaSiCPy.

Reference: Peng, T. et al. "BaSiC: a background and shading correction
method for optical microscopy images." Nature Communications 8, 14836
(2017), DOI 10.1038/s41467-017-01613-0. Upstream BaSiCPy:
https://github.com/peng-lab/BaSiCPy (MIT). This file is the lab PyTorch
adaptation BaSiCpy_torch, vendored into momia3; see README.md and LICENSE.
The conversion was assisted by doubao-seed-2.0-Code.

Key changes from JAX to PyTorch:
- Replaced JAX array operations with PyTorch tensor operations
- Added device detection and selection (CPU/CUDA/MPS)
- Replaced JAX jit with PyTorch eager execution
- Added PyTorch native resize support
- Maintained the same API for compatibility
"""

# Core modules
from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Suppress MPS fallback warnings
import warnings
warnings.filterwarnings("ignore", message="The operator 'aten::linalg_svd' is not currently supported on the MPS backend")
warnings.filterwarnings("ignore", message="The operator 'aten::.*' is not currently supported on the MPS backend")

# 3rd party modules
import numpy as np
import torch
from pydantic import BaseModel, Field, PrivateAttr, root_validator
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_erosion
from skimage.transform import resize as skimage_resize

# Optional hyperactive dependency for autotune feature
try:
    from hyperactive.base import _experiment as hyperactive
    from hyperactive.opt.gfo import HillClimbing
    HYPERACTIVE_AVAILABLE = True
except ImportError:
    HYPERACTIVE_AVAILABLE = False

from ._torch_routines import ApproximateFit, LadmapFit
from .metrics import autotune_cost
from .tools.dct_tools import TorchDCT

# Package modules
from .types import ArrayLike, PathLike

# Get number of available threads to limit CPU thrashing
# From preadator: https://pypi.org/project/preadator/
if hasattr(os, "sched_getaffinity"):
    # On Linux, we can detect how many cores are assigned to this process.
    # This is especially useful when running in a Docker container, when the
    # number of cores is intentionally limited.
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    # Default back to multiprocessing cpu_count, which is always going to count
    # the total number of cpus
    NUM_THREADS = cpu_count()

# initialize logger with the package name
logger = logging.getLogger(__name__)


class FittingMode(str, Enum):
    """Fit method enum."""

    ladmap: str = "ladmap"
    approximate: str = "approximate"


class ResizeMode(str, Enum):
    """Resize method enum."""

    jax: str = "jax"
    skimage: str = "skimage"
    skimage_dask: str = "skimage_dask"
    pytorch: str = "pytorch"


class TimelapseTransformMode(str, Enum):
    """Timelapse transformation enum."""

    additive: str = "additive"
    multiplicative: str = "multiplicative"


class Device(str, Enum):
    """Device to run on enum."""

    cpu: str = "cpu"
    cuda: str = "cuda"
    mps: str = "mps"
    tpu: str = "tpu"


_SETTINGS_FNAME = "settings.json"
_PROFILES_FNAME = "profiles.npz"


# multiple channels should be handled by creating a `basic` object for each channel
class BaSiC(BaseModel):
    """A class for fitting and applying BaSiC illumination correction profiles."""

    baseline: Optional[np.ndarray] = Field(
        None,
        description="Holds the baseline for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    darkfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the darkfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    fitting_mode: FittingMode = Field(
        FittingMode.ladmap, description="Must be one of ['ladmap', 'approximate']"
    )
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    flatfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the flatfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    smoothness_flatfield: float = Field(
        1.0, description="Weight of the flatfield term in the Lagrangian."
    )
    smoothness_darkfield: float = Field(
        1.0, description="Weight of the darkfield term in the Lagrangian."
    )
    sparse_cost_darkfield: float = Field(
        0.01, description="Weight of the darkfield sparse term in the Lagrangian."
    )
    autosegment: bool = Field(
        False,
        description="When not False, automatically segment the image before fitting."
        "When True, `threshold_otsu` from `scikit-image` is used "
        "and the brighter pixels are taken."
        "When a callable is given, it is used as the segmentation function.",
    )
    autosegment_margin: int = Field(
        10,
        description="Margin of the segmentation mask to the thresholded region.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    max_reweight_iterations: int = Field(
        10,
        description="Maximum number of reweighting iterations.",
    )
    max_reweight_iterations_baseline: int = Field(
        5,
        description="Maximum number of reweighting iterations for baseline.",
    )
    max_workers: int = Field(
        NUM_THREADS,
        description="Maximum number of threads used for processing.",
        exclude=True,  # Don't dump to output json/yaml
    )
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-3,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-2,
        description="Optimization tolerance for update diff.",
    )
    resize_mode: ResizeMode = Field(
        ResizeMode.pytorch,
        description="Resize mode to use when downsampling images. "
        + "Must be one of 'jax', 'skimage', 'skimage_dask', and 'pytorch'",
    )
    resize_params: Dict = Field(
        {},
        description="Parameters for the resize function when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Whether or not to sort the intensities of the image.",
    )
    working_size: Optional[Union[int, List[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )
    progress_callback: Optional[Callable[[float], None]] = Field(
        None,
        description="Callback function for progress reporting. Accepts a float between 0 and 1.",
        exclude=True,  # Don't dump to output json/yaml
    )
    device: Union[str, Device] = Field(
        "auto",
        description="Device to run on. Can be 'auto', 'cpu', 'cuda', 'mps', or 'tpu'. 'auto' will automatically detect available devices.",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _weight_dark: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)
    _smoothness_flatfield: float = PrivateAttr(None)
    _smoothness_darkfield: float = PrivateAttr(None)
    _sparse_cost_darkfield: float = PrivateAttr(None)
    _device: torch.device = PrivateAttr(None)

    class Config:
        """Pydantic class configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    @root_validator(pre=True)
    def debug_log_values(cls, values: Dict[str, Any]):
        """Use a validator to echo input values."""
        logger.debug("Initializing BaSiC with parameters:")
        for k, v in values.items():
            logger.debug(f"{k}: {v}")
        return values
    
    def __init__(self, **data):
        """Initialize the BaSiC object and set the device."""
        super().__init__(**data)
        
        # Device detection and setting
        device_param = self.device.lower()
        
        if device_param == "auto":
            # Auto-detect available devices
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU device")
        elif device_param in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self._device = torch.device("cpu")
        elif device_param == "mps":
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                logger.warning("MPS not available, falling back to CPU")
                self._device = torch.device("cpu")
        elif device_param == "tpu":
            logger.warning("TPU support is not implemented yet, falling back to CPU")
            self._device = torch.device("cpu")
        else:
            # Default to CPU
            self._device = torch.device("cpu")
            logger.info("Using CPU device")

    def __call__(
        self, images: np.ndarray, timelapse: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for `BaSiC.transform`."""
        return self.transform(images, timelapse)

    def _resize(self, Im, target_shape):
        if self.resize_mode == ResizeMode.pytorch:
            # Use PyTorch's native resize for better performance on GPU
            import torchvision.transforms as transforms
            
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(Im, np.ndarray):
                Im = torch.tensor(Im, dtype=torch.float32, device=self._device)
            
            # MPS does not implement antialiased upsample ops (e.g.
            # aten::_upsample_bilinear2d_aa.out). Fall back to CPU for the resize
            # and move the result back to the original device.
            original_device = Im.device
            if original_device.type == "mps":
                Im = Im.cpu()
            
            # Get target spatial dimensions
            spatial_dims = target_shape[-2:]
            
            # Resize each 2D image in the batch
            if Im.ndim == 3:  # (T, H, W)
                resized_images = []
                resize_transform = transforms.Resize(spatial_dims, **self.resize_params)
                for img in Im:
                    # Add channel dimension for transforms.Resize
                    img_reshaped = img.unsqueeze(0)
                    resized = resize_transform(img_reshaped)
                    resized_images.append(resized.squeeze(0))
                result = torch.stack(resized_images)
            elif Im.ndim == 4:  # (T, Z, H, W)
                resized_volumes = []
                resize_transform = transforms.Resize(spatial_dims, **self.resize_params)
                for volume in Im:
                    resized_slices = []
                    for slice_img in volume:
                        # Add channel dimension for transforms.Resize
                        img_reshaped = slice_img.unsqueeze(0)
                        resized = resize_transform(img_reshaped)
                        resized_slices.append(resized.squeeze(0))
                    resized_volumes.append(torch.stack(resized_slices))
                result = torch.stack(resized_volumes)
            else:
                raise ValueError(f"Unsupported image dimension: {Im.ndim}")
            
            return result.to(original_device)
        else:
            # Convert PyTorch tensor to numpy array if needed, ensuring it's on CPU first
            if isinstance(Im, torch.Tensor):
                Im = Im.cpu().numpy()
            
            if self.resize_mode == ResizeMode.jax:
                # PyTorch doesn't have a direct equivalent of jax.image.resize with ResizeMethod
                # We'll use skimage_resize instead for consistency
                Im = skimage_resize(
                    np.array(Im), target_shape, preserve_range=True, **self.resize_params
                )
                return torch.tensor(Im, dtype=torch.float32, device=self._device)

            elif self.resize_mode == ResizeMode.skimage:
                Im = skimage_resize(
                    np.array(Im), target_shape, preserve_range=True, **self.resize_params
                )
                return torch.tensor(Im, dtype=torch.float32, device=self._device)

            elif self.resize_mode == ResizeMode.skimage_dask:
                assert np.array_equal(target_shape[:-2], Im.shape[:-2])
                import dask.array as da

                Im = (
                    da.from_array(
                        [
                            skimage_resize(
                                np.array(Im[tuple(inds)]),
                                target_shape[-2:],
                                preserve_range=True,
                                **self.resize_params,
                            )
                            for inds in np.ndindex(Im.shape[:-2])
                        ]
                    )
                    .reshape((*Im.shape[:-2], *target_shape[-2:]))
                    .compute()
                )
                return torch.tensor(Im, dtype=torch.float32, device=self._device)

    def _resize_to_working_size(self, Im):
        """Resize the images to the working size."""
        # Convert numpy array to PyTorch tensor if needed
        if isinstance(Im, np.ndarray):
            Im = torch.tensor(Im, dtype=torch.float32, device=self._device)
        
        if self.working_size is not None:
            if np.isscalar(self.working_size):
                working_shape = [self.working_size] * (Im.ndim - 2)
            else:
                if not Im.ndim - 2 == len(self.working_size):
                    raise ValueError(
                        "working_size must be a scalar or match the image dimensions"
                    )
                else:
                    working_shape = self.working_size
            target_shape = [*Im.shape[:2], *working_shape]
            Im = self._resize(Im, target_shape)

        return Im

    def _perform_segmentation(self, Im):
        """Perform segmentation on the images."""
        # Convert PyTorch tensor to numpy array if needed
        if isinstance(Im, torch.Tensor):
            Im_np = Im.cpu().numpy()
        else:
            Im_np = np.array(Im)
        
        if not self.autosegment:
            return np.ones_like(Im_np, dtype=bool)
        elif self.autosegment is True:
            th = threshold_otsu(Im_np)
            mask = Im_np < th
            return np.array(
                [binary_erosion(m, ball(self.autosegment_margin)) for m in mask]
            )
        else:
            # If a custom autosegment function is provided, pass the original Im
            # It's up to the user to handle PyTorch tensors in their custom function
            return self.autosegment(Im)

    def fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
    ) -> None:
        """Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional or 4-dimensional array
                    with dimension of (T,Y,X) or (T,Z,Y,X).
                    T can be either of time or mosaic position.
                    Multichannel images should be
                    independently corrected for each channel.
            fitting_weight: Relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy import datasets as bdata
            >>> images = bdata.wsi_brain()
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, np.newaxis, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, np.newaxis, ...]
        elif images.ndim == 4:
            if self.fitting_mode == FittingMode.approximate:
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )

        if images.shape[-1] < 10 and not skip_shape_warning:
            logger.warning(
                "Image last dimension is less than 10. "
                + "Are you supplying images with the channel dimension?"
                + "Multichannel images should be "
                + "independently corrected for each channel."
            )

        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()

        Im = self._resize_to_working_size(images)

        if fitting_weight is not None:
            Ws = torch.tensor(fitting_weight, dtype=torch.float32, device=self._device)
            Ws = self._resize_to_working_size(Ws)
            # normalize relative weight to 0 to 1
            Ws_min = torch.min(Ws)
            Ws_max = torch.max(Ws)
            Ws = (Ws - Ws_min) / (Ws_max - Ws_min)
        else:
            Ws = torch.ones_like(Im, device=self._device)

        Ws = Ws * torch.tensor(self._perform_segmentation(Im), dtype=torch.float32, device=self._device)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = torch.argsort(Im, axis=0)
            Im2 = torch.take_along_dim(Im, inds, axis=0)
            Ws2 = torch.take_along_dim(Ws, inds, axis=0)
        else:
            Im2 = Im
            Ws2 = Ws

        if self.fitting_mode == FittingMode.approximate:
            mean_image = torch.mean(Im2, axis=0)
            mean_image = mean_image / torch.mean(Im2)
            mean_image_dct = TorchDCT.dct3d(mean_image.permute(*torch.arange(mean_image.ndim-1, -1, -1)))
            self._smoothness_flatfield = (
                torch.sum(torch.abs(mean_image_dct)) / 800 * self.smoothness_flatfield
            )
            self._smoothness_darkfield = (
                self._smoothness_flatfield * self.smoothness_darkfield / 2.5
            )
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
            self._smoothness_darkfield = self.smoothness_darkfield
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        logger.debug(f"_smoothness_flatfield set to {self._smoothness_flatfield}")
        logger.debug(f"_smoothness_darkfield set to {self._smoothness_darkfield}")
        logger.debug(f"_sparse_cost_darkfield set to {self._sparse_cost_darkfield}")

        # spectral_norm = torch.linalg.norm(Im.reshape((Im.shape[0], -1)), ord=2)
        _temp = torch.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp.S[0]

        if self.fitting_mode == FittingMode.approximate:
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im2.shape)
        fit_params = self.dict()
        fit_params.update(
            dict(
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                # matrix 2-norm (largest sing. value)
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=torch.min(Im2),
                image_norm=torch.linalg.norm(Im2.flatten(), ord=2),
            )
        )

        # Initialize variables
        W = torch.ones_like(Im2, dtype=torch.float32, device=self._device) * Ws2
        W_D = torch.ones(Im2.shape[1:], dtype=torch.float32, device=self._device)
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == FittingMode.ladmap:
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        # Set up progress tracking
        progress_callback = self.progress_callback
        tqdm_bar = None
        total_iterations = self.max_reweight_iterations + self.max_reweight_iterations_baseline
        current_iteration = 0
        
        # Use tqdm if no callback is provided and tqdm is available
        if progress_callback is None:
            try:
                from tqdm.auto import tqdm
                tqdm_bar = tqdm(total=100, desc="BaSiC Fitting", unit="%")
                
                def tqdm_progress(progress):
                    if tqdm_bar:
                        tqdm_bar.update(int(progress * 100) - tqdm_bar.n)
                
                progress_callback = tqdm_progress
            except ImportError:
                # tqdm not available, leave progress_callback as None
                pass

        try:
            for i in range(self.max_reweight_iterations):
                logger.debug(f"reweighting iteration {i}")
                if self.fitting_mode == FittingMode.approximate:
                    S = torch.zeros(Im2.shape[1:], dtype=torch.float32, device=self._device)
                else:
                    S = torch.median(Im2, axis=0).values.to(self._device)
                D_R = torch.zeros(Im2.shape[1:], dtype=torch.float32, device=self._device)
                D_Z = torch.tensor(0.0, dtype=torch.float32, device=self._device)
                if self.fitting_mode == FittingMode.approximate:
                    B = torch.ones(Im2.shape[0], dtype=torch.float32, device=self._device)
                else:
                    B = torch.ones(Im2.shape[0], dtype=torch.float32, device=self._device)
                I_R = torch.zeros(Im2.shape, dtype=torch.float32, device=self._device)
                S, D_R, D_Z, I_R, B, norm_ratio, converged = fitting_step.fit(
                    Im2,
                    W,
                    W_D,
                    S,
                    D_R,
                    D_Z,
                    B,
                    I_R,
                    progress_callback=None,  # Don't pass callback to inner loop
                )
                logger.debug(f"single-step optimization score: {norm_ratio}.")
                logger.debug(f"mean of S: {float(torch.mean(S))}.")
                self._score = norm_ratio
                if not converged:
                    logger.debug("single-step optimization did not converge.")
                if torch.max(S) == 0:
                    logger.error(
                        "Estimated flatfield is zero. "
                        + "Please try to decrease smoothness_darkfield."
                    )
                    raise RuntimeError(
                        "Estimated flatfield is zero. "
                        + "Please try to decrease smoothness_darkfield."
                    )
                self._S = S
                self._D_R = D_R
                self._B = B
                self._D_Z = D_Z
                D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
                mean_S = torch.mean(S)
                S = S / mean_S  # flatfields
                B = B * mean_S  # baseline
                I_B = B.view(-1, 1, 1, 1) * S.unsqueeze(0) + D.unsqueeze(0)
                W = fitting_step.calc_weights(I_B, I_R) * Ws2
                W_D = fitting_step.calc_dark_weights(D_R)

                self._weight = W
                self._weight_dark = W_D
                self._residual = I_R

                logger.debug(f"Iteration {i} finished.")
                
                # Update progress
                current_iteration += 1
                if progress_callback is not None:
                    progress = current_iteration / total_iterations
                    progress_callback(progress)
                
                if last_S is not None:
                    mad_flatfield = torch.sum(torch.abs(S - last_S)) / torch.sum(torch.abs(last_S))
                    if self.get_darkfield:
                        mad_darkfield = torch.sum(torch.abs(D - last_D)) / max(
                            torch.sum(torch.abs(last_D)), 1
                        )  # assumes the amplitude of darkfield is more than 1
                        self._reweight_score = max(mad_flatfield, mad_darkfield)
                    else:
                        self._reweight_score = mad_flatfield
                    logger.debug(f"reweighting score: {self._reweight_score}")
                    logger.info(
                        f"Iteration {i} elapsed time: "
                        + f"{time.monotonic() - start_time} seconds"
                    )

                    if self._reweight_score <= self.reweighting_tol:
                        logger.info("Reweighting converged.")
                        break
                if i == self.max_reweight_iterations - 1:
                    logger.warning("Reweighting did not converge.")
                last_S = S
                last_D = D

        finally:
            # Always ensure we reach 100% progress when done
            if progress_callback is not None:
                progress_callback(1.0)
            
            # Clean up tqdm bar if it exists
            if tqdm_bar is not None:
                tqdm_bar.close()

        if not converged:
            logger.warning(
                "Single-step optimization did not converge "
                + "at the last reweighting step."
            )

        assert S is not None
        assert D is not None
        assert B is not None

        # Set up progress tracking for baseline reweighting
        if self.sort_intensity:
            for i in range(self.max_reweight_iterations_baseline):
                B = torch.ones(Im.shape[0], dtype=torch.float32, device=self._device)
                if self.fitting_mode == FittingMode.approximate:
                    B = torch.mean(Im, axis=(1, 2, 3))
                I_R = torch.zeros(Im.shape, dtype=torch.float32, device=self._device)
                logger.debug(f"reweighting iteration for baseline {i}")
                I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
                    Im,
                    W,
                    S,
                    D,
                    B,
                    I_R,
                    progress_callback=None,  # Don't pass callback to inner loop
                )

                I_B = B.view(-1, 1, 1, 1) * S.unsqueeze(0) + D.unsqueeze(0)
                W = fitting_step.calc_weights_baseline(I_B, I_R) * Ws
                self._weight = W
                self._residual = I_R
                logger.debug(f"baseline iteration {i} finished.")
                
                # Update progress
                current_iteration += 1
                if progress_callback is not None:
                    progress = current_iteration / total_iterations
                    progress_callback(progress)

        # Final progress update to ensure we reach 100%
        if progress_callback is not None:
            progress_callback(1.0)

        self.flatfield = skimage_resize(S.cpu().numpy(), images.shape[1:])
        self.darkfield = skimage_resize(D.cpu().numpy(), images.shape[1:])
        if ndim == 3:
            self.flatfield = self.flatfield[0]
            self.darkfield = self.darkfield[0]
        self.baseline = B.cpu().numpy()
        logger.info(
            f"=== BaSiC fit finished in {time.monotonic()-start_time} seconds ==="
        )

    def transform(
        self,
        images: np.ndarray,
        timelapse: Union[bool, TimelapseTransformMode] = False,
        frames: Optional[Sequence[Union[int, np.int_]]] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Apply profile to images.

        Args:
            images: input images to correct. See `fit`.
            timelapse: If `True`, corrects the timelapse/photobleaching offsets,
                       assuming that the residual is the product of flatfield and
                       the object fluorescence. Also accepts "multiplicative"
                       (the same as `True`) or "additive" (residual is the object
                       fluorescence).
            frames: Frames to use for transformation. Defaults to None (all frames).

        Returns:
            corrected images

        Example:
            >>> basic.fit(images)
            >>> corrected = basic.transform(images)
        """
        if self.baseline is None:
            raise RuntimeError("BaSiC object is not initialized")

        logger.info("=== BaSiC transform started ===")
        start_time = time.monotonic()

        # Convert to the correct format
        im_float = images.astype(np.float32)

        # Image = B_n x S_l + D_l + I_R_nl

        # in timelapse cases ...
        # "Multiplicative" mode
        # Real Image x S_l = I_R_nl
        # Image = (B_n + Real Image) x S_l + D_l
        # Real Image = (Image - D_l) / S_l - B_n

        # "Additive" mode
        # Real Image = I_R_nl
        # Image = B_n x S_l + D_l + Real Image
        # Real Image = Image - D_l - (S_l x B_n)

        # in non-timelapse cases ...
        # we assume B_n is the mean of Real Image
        # and then always assume Multiplicative mode
        # the image model is
        # Image = Real Image x S_l + D_l
        # Real Image = (Image - D_l) / S_l

        if timelapse:
            if timelapse is True:
                timelapse = TimelapseTransformMode.multiplicative
            if frames is None:
                _frames = slice(None)
            else:
                _frames = np.array(frames)
            baseline_inds = tuple([_frames] + ([np.newaxis] * (im_float.ndim - 1)))
            if timelapse == TimelapseTransformMode.multiplicative:
                output = (im_float - self.darkfield[np.newaxis]) / self.flatfield[
                    np.newaxis
                ] - self.baseline[baseline_inds]
            elif timelapse == TimelapseTransformMode.additive:
                baseline_flatfield = (
                    self.flatfield[np.newaxis] * self.baseline[baseline_inds]
                )
                output = im_float - self.darkfield[np.newaxis] - baseline_flatfield
            else:
                raise ValueError(
                    "timelapse value must be bool, 'multiplicative' or 'additive'"
                )
        else:
            output = (im_float - self.darkfield[np.newaxis]) / self.flatfield[
                np.newaxis
            ]
        logger.info(
            f"=== BaSiC transform finished in {time.monotonic()-start_time} seconds ==="
        )
        return output

    # REFACTOR large datasets will probably prefer saving corrected images to
    # files directly, a generator may be handy
    def fit_transform(
        self,
        images: ArrayLike,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
        timelapse: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit and transform on data.

        Args:
            images: input images to fit and correct. See `fit`.

        Returns:
            corrected images

        Example:
            >>> corrected = basic.fit_transform(images)
        """
        self.fit(
            images, fitting_weight=fitting_weight, skip_shape_warning=skip_shape_warning
        )
        corrected = self.transform(images, timelapse)

        return corrected

    def autotune(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning: bool = False,
        *,
        optmizer=None,
        n_iter=100,
        search_space=None,
        init_params=None,
        timelapse: bool = False,
        histogram_qmin: float = 0.01,
        histogram_qmax: float = 0.99,
        vmin_factor: float = 0.6,
        vrange_factor: float = 1.5,
        histogram_bins: int = 1000,
        histogram_use_fitting_weight: bool = True,
        fourier_l0_norm_image_threshold: float = 0.1,
        fourier_l0_norm_fourier_radius=10,
        fourier_l0_norm_threshold=0.0,
        fourier_l0_norm_cost_coef=30,
        early_stop: bool = True,
        early_stop_n_iter_no_change: int = 15,
        early_stop_torelance: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        """Automatically tune the parameters of the model.

        Args:
            images: input images to fit and correct. See `fit`.
            fitting_weight: Relative fitting weight for each pixel. See `fit`.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.
            optimizer: optimizer to use. Defaults to
                    `hyperactive.optimizers.HillClimbingOptimizer`.
            n_iter: number of iterations for the optimizer. Defaults to 100.
            search_space: search space for the optimizer.
                    Defaults to a reasonable range for each parameter.
            init_params: initial parameters for the optimizer.
                    Defaults to a reasonable initial value for each parameter.
            timelapse: if True, corrects the timelapse/photobleaching offsets.
            histogram_qmin: the minimum quantile to use for the histogram.
                    Defaults to 0.01.
            histogram_qmax: the maximum quantile to use for the histogram.
                    Defaults to 0.99.
            histogram_bins: the number of bins to use for the histogram.
                    Defaults to 100.
            hisogram_use_fitting_weight: if True, uses the weight for the histogram.
                    Defaults to True.
            fourier_l0_norm_image_threshold : float
                The threshold for image values for the fourier L0 norm calculation.
            fourier_l0_norm_fourier_radius : float
                The Fourier radius for the fourier L0 norm calculation.
            fourier_l0_norm_threshold : float
                The maximum preferred value for the fourier L0 norm.
            fourier_l0_norm_cost_coef : float
                The cost coefficient for the fourier L0 norm.
            early_stop: if True, stops the optimization when the change in
                    entropy is less than `early_stop_torelance`.
                    Defaults to True.
            early_stop_n_iter_no_change: the number of iterations for early
                    stopping. Defaults to 10.
            early_stop_torelance: the absolute value torelance
                    for early stopping.
            random_state: random state for the optimizer.

        """
        if not HYPERACTIVE_AVAILABLE:
            raise ImportError(
                "hyperactive package is required for autotune feature. "
                "Please install it with: pip install hyperactive"
            )

        if search_space is None:
            search_space = {
                "smoothness_flatfield": list(np.logspace(-3, 1, 15)),
            }
            if self.get_darkfield:
                search_space.update(
                    {
                        "smoothness_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                        "sparse_cost_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                    }
                )
        if init_params is None:
            init_params = {
                "smoothness_flatfield": 0.1,
            }
            if self.get_darkfield:
                init_params.update(
                    {
                        "smoothness_darkfield": 1e-3,
                        "sparse_cost_darkfield": 1e-3,
                    }
                )

        # calculate the histogram range
        basic = self.copy(update=init_params)
        basic.fit(
            images,
            fitting_weight=fitting_weight,
            skip_shape_warning=skip_shape_warning,
        )
        transformed = basic.transform(images, timelapse=timelapse)
        vmin, vmax = np.quantile(transformed, [histogram_qmin, histogram_qmax])
        val_range = (
            vmax - vmin * vmin_factor
        ) * vrange_factor  # fix the value range for histogram

        if fitting_weight is None or not histogram_use_fitting_weight:
            weights = None
        else:
            weights = fitting_weight

        def fit_and_calc_entropy(params):
            try:
                basic = self.copy(update=params)
                basic.fit(
                    images,
                    fitting_weight=fitting_weight,
                    skip_shape_warning=skip_shape_warning,
                )
                transformed = basic.transform(images, timelapse=timelapse)
                vmin_new = np.quantile(transformed, histogram_qmin) * vmin_factor

                if np.allclose(basic.flatfield, np.ones_like(basic.flatfield)):
                    return -np.inf  # discard the case where flatfield is all ones

                return -1.0 * autotune_cost(
                    transformed,
                    basic.flatfield,
                    entropy_vmin=vmin_new,
                    entropy_vmax=vmin_new + val_range,
                    histogram_bins=histogram_bins,
                    fourier_l0_norm_cost_coef=fourier_l0_norm_cost_coef,
                    fourier_l0_norm_image_threshold=fourier_l0_norm_image_threshold,
                    fourier_l0_norm_fourier_radius=fourier_l0_norm_fourier_radius,
                    fourier_l0_norm_threshold=fourier_l0_norm_threshold,
                    weights=weights,
                )
            except RuntimeError:
                return -np.inf

        if optmizer is None:
            optimizer = HillClimbing(
                epsilon=0.1,
                distribution="laplace",
                n_neighbours=4,
                rand_rest_p=0.1,
            )

        hyper = hyperactive()

        params = dict(
            optimizer=optimizer,
            n_iter=n_iter,
            initialize=dict(warm_start=[init_params]),
            random_state=random_state,
        )

        if early_stop:
            params.update(
                dict(
                    early_stopping=dict(
                        n_iter_no_change=early_stop_n_iter_no_change,
                        tol_abs=early_stop_torelance,
                    )
                )
            )

        hyper.add_search(
            fit_and_calc_entropy,
            search_space,
            **params,
        )
        hyper.run()
        best_params = hyper.best_para(fit_and_calc_entropy)
        self.__dict__.update(best_params)

    @property
    def score(self):
        """The BaSiC fit final score."""
        return self._score

    @property
    def reweight_score(self):
        """The BaSiC fit final reweighting score."""
        return self._reweight_score

    @property
    def settings(self) -> Dict:
        """Current settings.

        Returns:
            current settings
        """
        return self.dict()

    def save_model(self, model_dir: PathLike, overwrite: bool = False) -> None:
        """Save current model to folder.

        Args:
            model_dir: path to model directory

        Raises:
            FileExistsError: if model directory already exists
        """
        path = Path(model_dir)

        try:
            path.mkdir()
        except FileExistsError:
            if not overwrite:
                raise FileExistsError("Model folder already exists.")

        # save settings
        with open(path / _SETTINGS_FNAME, "w") as fp:
            # see pydantic docs for output options
            fp.write(self.json())

        # NOTE emit warning if profiles are all zeros? fit probably not run
        # save profiles
        np.savez(
            path / _PROFILES_FNAME,
            flatfield=np.array(self.flatfield),
            darkfield=np.array(self.darkfield),
            baseline=np.array(self.baseline),
        )

    @classmethod
    def load_model(cls, model_dir: PathLike) -> BaSiC:
        """Create a new instance from a model folder."""
        path = Path(model_dir)

        if not path.exists():
            raise FileNotFoundError("Model directory not found.")

        with open(path / _SETTINGS_FNAME) as fp:
            model = json.load(fp)

        profiles = np.load(path / _PROFILES_FNAME)
        model["flatfield"] = profiles["flatfield"]
        model["darkfield"] = profiles["darkfield"]
        model["baseline"] = profiles["baseline"]

        return BaSiC(**model)
