"""OmniPose-native flow/distance/boundary targets and vector-safe augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from momia3.models.probability.data import AugmentParams, Sample, normalize_image


def labels_to_omni_target(labels: np.ndarray) -> np.ndarray:
    """Return ``(flow_y, flow_x, distance, boundary, weight)`` using OmniPose."""
    from omnipose.core import batch_labels, masks_to_flows_batch

    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError(f"expected 2D labels, got {labels.shape}")
    masks, boundary, distance, flow, slices, _links, _coords, _affinity = (
        masks_to_flows_batch(
            labels[None], links=[None], device=torch.device("cpu"), omni=True
        )
    )
    masks, boundary, distance, flow = [
        torch.stack([x[(Ellipsis,) + slc] for slc in slices])
        for x in (masks, boundary, distance, flow)
    ]
    packed = batch_labels(
        masks, boundary, distance, flow, labels.shape,
        dim=2, nclasses=4, device=torch.device("cpu")
    )
    target = packed[0, [5, 6, 3, 2, 4]].numpy().astype(np.float32)
    if not np.all(np.isfinite(target)):
        raise ValueError("non-finite OmniPose target")
    return target


def cached_omni_target(sample: Sample, cache_dir: str | Path) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sample.sample_id.replace('/', '__')}.npy"
    if path.exists():
        return np.load(path)
    target = labels_to_omni_target(tifffile.imread(sample.mask))
    np.save(path, target)
    return target


def _crop_field(
    image: np.ndarray, target: np.ndarray, size: int, rng: np.random.Generator
):
    h, w = image.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h or pad_w:
        pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        image = np.pad(image, pads, mode="reflect")
        target = np.pad(target, ((0, 0),) + pads, mode="constant")
        target[2, : pad_h // 2] = -5.0
        target[2, - (pad_h - pad_h // 2) :] = -5.0 if pad_h else target[2, -0:]
        target[2, :, : pad_w // 2] = -5.0
        target[2, :, - (pad_w - pad_w // 2) :] = -5.0 if pad_w else target[2, :, -0:]
    h, w = image.shape
    foreground = np.flatnonzero(target[4].ravel() > 0.5)
    if foreground.size and rng.random() < 0.7:
        cy, cx = divmod(int(rng.choice(foreground)), w)
        jitter = max(1, size // 4)
        y = cy - size // 2 + int(rng.integers(-jitter, jitter + 1))
        x = cx - size // 2 + int(rng.integers(-jitter, jitter + 1))
    else:
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
    y = int(np.clip(y, 0, h - size))
    x = int(np.clip(x, 0, w - size))
    return (
        image[y : y + size, x : x + size],
        target[:, y : y + size, x : x + size],
    )


def _fit_field_shape(image: np.ndarray, target: np.ndarray, shape: tuple[int, int]):
    ys = max(0, (image.shape[0] - shape[0]) // 2)
    xs = max(0, (image.shape[1] - shape[1]) // 2)
    image = image[ys : ys + shape[0], xs : xs + shape[1]]
    target = target[:, ys : ys + shape[0], xs : xs + shape[1]]
    pad_h = shape[0] - image.shape[0]
    pad_w = shape[1] - image.shape[1]
    if pad_h or pad_w:
        pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        image = np.pad(image, pads, mode="reflect")
        target = np.pad(target, ((0, 0),) + pads, mode="constant")
        valid = np.zeros(shape, dtype=bool)
        valid[pads[0][0] : pads[0][0] + target.shape[1] - pad_h,
              pads[1][0] : pads[1][0] + target.shape[2] - pad_w] = True
        target[2, ~valid] = -5.0
    return image, target


def _rotate_field(image: np.ndarray, target: np.ndarray, angle: float):
    image = ndimage.rotate(image, angle, reshape=False, order=1, mode="reflect")
    rotated = np.stack([
        ndimage.rotate(
            channel, angle, reshape=False, order=1, mode="constant",
            cval=-5.0 if index == 2 else 0.0,
        )
        for index, channel in enumerate(target)
    ])
    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    flow_y = rotated[0].copy()
    flow_x = rotated[1].copy()
    rotated[0] = cosine * flow_y - sine * flow_x
    rotated[1] = sine * flow_y + cosine * flow_x
    return image, rotated


def _scale_field(image: np.ndarray, target: np.ndarray, scale: float):
    original_shape = image.shape
    image = ndimage.zoom(image, scale, order=1, mode="reflect")
    target = np.stack([
        ndimage.zoom(
            channel, scale, order=1, mode="constant",
            cval=-5.0 if index == 2 else 0.0,
        )
        for index, channel in enumerate(target)
    ])
    target[:2] *= scale
    target[2] = np.where(target[2] <= -5.0 + 1e-6, -5.0, target[2] * scale)
    return _fit_field_shape(image, target, original_shape)


def random_field_augment(
    image: np.ndarray, target: np.ndarray, rng: np.random.Generator,
    patch_size: int = 256, params: AugmentParams | None = None,
):
    """Apply synchronized spatial and photometric augmentation to field targets."""
    params = params or AugmentParams(patch_size=patch_size)
    image = np.array(image, dtype=np.float32, copy=True)
    target = np.array(target, dtype=np.float32, copy=True)
    image, target = _crop_field(image, target, params.patch_size, rng)
    if rng.random() < params.rotate_probability:
        image, target = _rotate_field(image, target, float(rng.uniform(0.0, 360.0)))
    if rng.random() < params.scale_probability:
        image, target = _scale_field(image, target, float(rng.uniform(*params.scale_range)))
    if rng.random() < 0.5:
        image = np.flip(image, 0)
        target = np.flip(target, 1)
        target[0] = -target[0]
    if rng.random() < 0.5:
        image = np.flip(image, 1)
        target = np.flip(target, 2)
        target[1] = -target[1]
    if rng.random() < params.contrast_probability:
        gamma = float(rng.uniform(*params.gamma_range))
        contrast = float(rng.uniform(*params.contrast_range))
        mean = float(image.mean())
        image = (np.clip(image, 0, 1) ** gamma - mean) * contrast + mean
    if rng.random() < params.blur_probability:
        image = ndimage.gaussian_filter(image, float(rng.uniform(*params.blur_sigma_range)))
    if rng.random() < params.noise_probability:
        if rng.random() < 0.5:
            image = image + rng.normal(
                0.0, float(rng.uniform(*params.gaussian_noise_range)), image.shape
            )
        else:
            photons = float(rng.uniform(*params.poisson_photons_range))
            image = rng.poisson(np.clip(image, 0, 1) * photons) / photons
    return (
        np.clip(image, 0, 1).astype(np.float32),
        np.ascontiguousarray(target, dtype=np.float32),
    )


class OmniPoseFieldDataset(Dataset):
    """Normalized images with cached OmniPose-native five-channel targets."""

    def __init__(
        self,
        samples: list[Sample],
        cache_dir: str | Path,
        training: bool,
        patch_size: int = 256,
        seed: int = 2026,
        augment: AugmentParams | None = None,
    ) -> None:
        self.samples = list(samples)
        self.training = training
        self.patch_size = patch_size
        self.augment = augment or AugmentParams(patch_size=patch_size)
        self.seed = seed
        self.images = [normalize_image(tifffile.imread(s.image)) for s in self.samples]
        self.targets = [cached_omni_target(s, cache_dir) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = self.images[index]
        target = self.targets[index]
        if self.training:
            draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
            image, target = random_field_augment(
                image, target, np.random.default_rng([self.seed, index, draw]),
                self.patch_size, self.augment,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(image))[None].float(),
            torch.from_numpy(np.ascontiguousarray(target)).float(),
        )
