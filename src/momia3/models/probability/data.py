"""OmniPose instance-mask data preparation for three-class probability models."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile
from pydantic import BaseModel, Field
from scipy import ndimage
from skimage import segmentation
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    group: str
    split: str = ""

    @property
    def sample_id(self) -> str:
        return f"{self.group}/{self.image.stem}"


class AugmentParams(BaseModel):
    patch_size: int = Field(256, ge=32)
    cell_centered_probability: float = Field(0.7, ge=0.0, le=1.0)
    rotate_probability: float = Field(0.8, ge=0.0, le=1.0)
    scale_probability: float = Field(0.3, ge=0.0, le=1.0)
    scale_range: tuple[float, float] = Field((0.9, 1.1))
    elastic_probability: float = Field(0.15, ge=0.0, le=1.0)
    elastic_alpha: float = Field(8.0, ge=0.0)
    elastic_sigma: float = Field(8.0, gt=0.0)
    contrast_probability: float = Field(0.5, ge=0.0, le=1.0)
    gamma_range: tuple[float, float] = Field((0.8, 1.25))
    contrast_range: tuple[float, float] = Field((0.85, 1.18))
    blur_probability: float = Field(0.2, ge=0.0, le=1.0)
    blur_sigma_range: tuple[float, float] = Field((0.3, 1.0))
    noise_probability: float = Field(0.3, ge=0.0, le=1.0)
    gaussian_noise_range: tuple[float, float] = Field((0.01, 0.05))
    poisson_photons_range: tuple[float, float] = Field((30.0, 100.0))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a full frame by its 1st/99th percentiles to float32 [0, 1]."""
    work = np.asarray(image, dtype=np.float32)
    if work.ndim != 2:
        raise ValueError(f"expected a 2D image, got {work.shape}")
    lo, hi = np.percentile(work, (1.0, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros(work.shape, dtype=np.float32)
    return np.clip((work - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def labels_to_target(labels: np.ndarray, boundary_width: int = 1) -> np.ndarray:
    """Convert instance labels to background/edge/interior class indices.

    Every retained instance has at least one interior pixel: for very small
    objects, the boundary target is removed rather than erasing the object.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected 2D instance labels, got {labels.shape}")
    if boundary_width < 1:
        raise ValueError("boundary_width must be >= 1")
    foreground = labels > 0
    edge = segmentation.find_boundaries(labels, mode="inner") & foreground
    if boundary_width > 1:
        from skimage import morphology

        edge = morphology.dilation(edge, morphology.disk(boundary_width - 1)) & foreground
    for label_id in np.unique(labels[foreground]):
        cell = labels == label_id
        if not np.any(cell & ~edge):
            edge[cell] = False
    target = np.zeros(labels.shape, dtype=np.uint8)
    target[foreground] = 2
    target[edge] = 1
    return target


def discover_samples(root: str | Path) -> list[Sample]:
    """Discover image/``_masks`` TIFF pairs, grouped by their source directory."""
    root = Path(root)
    samples = []
    for image in sorted(root.rglob("*.tif")):
        if image.stem.endswith("_masks"):
            continue
        mask = image.with_name(f"{image.stem}_masks.tif")
        if not mask.exists():
            continue
        samples.append(Sample(image, mask, image.parent.relative_to(root).as_posix()))
    if not samples:
        raise ValueError(f"no OmniPose image/mask pairs found under {root}")
    return samples


def split_samples(
    samples: list[Sample], seed: int = 2026, ratios=(0.70, 0.15, 0.15)
) -> list[Sample]:
    """Greedily balance whole source groups across train/validation/test splits."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise ValueError("ratios must contain three positive values")
    ratio_sum = float(sum(ratios))
    ratios = tuple(float(r) / ratio_sum for r in ratios)
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.group, []).append(sample)
    if len(grouped) < 3:
        raise ValueError("group-aware splitting requires at least three source groups")

    rng = np.random.default_rng(seed)
    tie_break = {name: float(rng.random()) for name in grouped}
    names = sorted(grouped, key=lambda n: (-len(grouped[n]), tie_break[n]))
    split_names = ("train", "val", "test")
    targets = np.asarray(ratios) * len(samples)
    counts = np.zeros(3, dtype=np.int64)
    assignment: dict[str, str] = {}
    for name in names:
        remaining = targets - counts
        index = int(np.argmax(remaining + rng.random(3) * 1e-6))
        assignment[name] = split_names[index]
        counts[index] += len(grouped[name])
    return [replace(sample, split=assignment[sample.group]) for sample in samples]


def write_split(
    path: str | Path, samples: list[Sample], root: str | Path, seed: int = 2026
) -> None:
    root = Path(root)
    payload = {
        "version": 1,
        "seed": seed,
        "root": str(root),
        "samples": [
            {
                "image": sample.image.relative_to(root).as_posix(),
                "mask": sample.mask.relative_to(root).as_posix(),
                "group": sample.group,
                "split": sample.split,
            }
            for sample in samples
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_split(path: str | Path, root: str | Path) -> list[Sample]:
    payload = json.loads(Path(path).read_text())
    root = Path(root)
    return [
        Sample(
            image=root / row["image"],
            mask=root / row["mask"],
            group=row["group"],
            split=row["split"],
        )
        for row in payload["samples"]
    ]


def load_or_create_split(
    path: str | Path, root: str | Path, seed: int = 2026
) -> list[Sample]:
    path = Path(path)
    if path.exists():
        return load_split(path, root)
    samples = split_samples(discover_samples(root), seed=seed)
    write_split(path, samples, root, seed=seed)
    return samples


def _fit_shape(image: np.ndarray, target: np.ndarray, shape: tuple[int, int]):
    ys = max(0, (image.shape[0] - shape[0]) // 2)
    xs = max(0, (image.shape[1] - shape[1]) // 2)
    image = image[ys : ys + shape[0], xs : xs + shape[1]]
    target = target[ys : ys + shape[0], xs : xs + shape[1]]
    pad_h = shape[0] - image.shape[0]
    pad_w = shape[1] - image.shape[1]
    if pad_h or pad_w:
        pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        image = np.pad(image, pads, mode="reflect")
        target = np.pad(target, pads, mode="constant")
    return image, target


def _crop_patch(
    image: np.ndarray, target: np.ndarray, size: int, rng: np.random.Generator,
    cell_centered_probability: float,
):
    if image.shape[0] < size or image.shape[1] < size:
        pad_h = max(0, size - image.shape[0])
        pad_w = max(0, size - image.shape[1])
        pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        image = np.pad(image, pads, mode="reflect")
        target = np.pad(target, pads, mode="constant")
    h, w = image.shape
    foreground = np.flatnonzero(target.ravel() > 0)
    if foreground.size and rng.random() < cell_centered_probability:
        cy, cx = divmod(int(rng.choice(foreground)), w)
        jitter = max(1, size // 4)
        y = cy - size // 2 + int(rng.integers(-jitter, jitter + 1))
        x = cx - size // 2 + int(rng.integers(-jitter, jitter + 1))
    else:
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
    y = int(np.clip(y, 0, h - size))
    x = int(np.clip(x, 0, w - size))
    return image[y : y + size, x : x + size], target[y : y + size, x : x + size]


def random_augment(
    image: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply synchronized geometry and mild photometric augmentation."""
    params = params or AugmentParams()
    image = np.asarray(image, dtype=np.float32).copy()
    target = np.asarray(target, dtype=np.uint8).copy()
    if rng.random() < params.rotate_probability:
        angle = float(rng.uniform(0.0, 360.0))
        image = ndimage.rotate(image, angle, reshape=False, order=1, mode="reflect")
        target = ndimage.rotate(target, angle, reshape=False, order=0, mode="constant")
    if rng.random() < params.scale_probability:
        scale = float(rng.uniform(*params.scale_range))
        original_shape = image.shape
        image = ndimage.zoom(image, scale, order=1, mode="reflect")
        target = ndimage.zoom(target, scale, order=0, mode="constant")
        image, target = _fit_shape(image, target, original_shape)
    if rng.random() < 0.5:
        image, target = np.flip(image, 0), np.flip(target, 0)
    if rng.random() < 0.5:
        image, target = np.flip(image, 1), np.flip(target, 1)
    if rng.random() < params.elastic_probability:
        yy, xx = np.meshgrid(
            np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij"
        )
        dy = ndimage.gaussian_filter(rng.normal(size=image.shape), params.elastic_sigma)
        dx = ndimage.gaussian_filter(rng.normal(size=image.shape), params.elastic_sigma)
        dy *= params.elastic_alpha / max(float(np.std(dy)), 1e-6)
        dx *= params.elastic_alpha / max(float(np.std(dx)), 1e-6)
        coords = (yy + dy, xx + dx)
        image = ndimage.map_coordinates(image, coords, order=1, mode="reflect")
        target = ndimage.map_coordinates(target, coords, order=0, mode="constant")

    image, target = _crop_patch(
        image, target, params.patch_size, rng, params.cell_centered_probability
    )
    if rng.random() < params.contrast_probability:
        gamma = float(rng.uniform(*params.gamma_range))
        contrast = float(rng.uniform(*params.contrast_range))
        mean = float(image.mean())
        image = (np.clip(image, 0.0, 1.0) ** gamma - mean) * contrast + mean
    if rng.random() < params.blur_probability:
        image = ndimage.gaussian_filter(
            image, sigma=float(rng.uniform(*params.blur_sigma_range))
        )
    if rng.random() < params.noise_probability:
        if rng.random() < 0.5:
            image = image + rng.normal(
                0.0, float(rng.uniform(*params.gaussian_noise_range)), image.shape
            )
        else:
            photons = float(rng.uniform(*params.poisson_photons_range))
            image = rng.poisson(np.clip(image, 0.0, 1.0) * photons) / photons
    return np.clip(image, 0.0, 1.0).astype(np.float32), target.astype(np.int64)


class OmniPoseDataset(Dataset):
    """In-memory normalized OmniPose frames with synchronized patch augmentation."""

    def __init__(
        self,
        samples: list[Sample],
        training: bool,
        augment: AugmentParams | None = None,
        boundary_width: int = 1,
        seed: int = 2026,
    ) -> None:
        self.samples = list(samples)
        self.training = training
        self.augment = augment or AugmentParams()
        self.seed = seed
        self.images = []
        self.targets = []
        for sample in self.samples:
            image = tifffile.imread(sample.image)
            labels = tifffile.imread(sample.mask)
            if image.shape != labels.shape:
                raise ValueError(f"image/mask shape mismatch for {sample.sample_id}")
            self.images.append(normalize_image(image))
            self.targets.append(labels_to_target(labels, boundary_width=boundary_width))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        image = self.images[index]
        target = self.targets[index]
        if self.training:
            draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = np.random.default_rng([self.seed, index, draw])
            image, target = random_augment(image, target, rng, self.augment)
        return (
            torch.from_numpy(np.ascontiguousarray(image))[None].float(),
            torch.from_numpy(np.ascontiguousarray(target)).long(),
        )


def build_expanded_dataset(
    original_root: str | Path,
    supplement_manifest: str | Path,
    output_root: str | Path,
) -> list[Sample]:
    """Copy original and QC-approved ROI pairs into one grouped dataset."""
    import shutil

    original_root = Path(original_root)
    output_root = Path(output_root)
    supplement = json.loads(Path(supplement_manifest).read_text())
    copied = []
    for sample in discover_samples(original_root):
        target_dir = output_root / sample.group
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / sample.image.name
        target_mask = target_dir / sample.mask.name
        shutil.copy2(sample.image, target_image)
        shutil.copy2(sample.mask, target_mask)
        copied.append({"source": "bact_phase", "group": sample.group, "image": sample.image.name})
    for row in supplement["images"]:
        if not row.get("include", True):
            continue
        group = f"selected_myco/{row['series']}"
        target_dir = output_root / group
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / Path(row["output_image"]).name
        target_mask = target_dir / Path(row["output_mask"]).name
        shutil.copy2(row["output_image"], target_image)
        shutil.copy2(row["output_mask"], target_mask)
        copied.append({"source": "selected_myco", "group": group, "image": target_image.name})
    samples = discover_samples(output_root)
    payload = {
        "version": 1,
        "original_root": str(original_root),
        "supplement_manifest": str(supplement_manifest),
        "n_original": sum(row["source"] == "bact_phase" for row in copied),
        "n_supplement": sum(row["source"] == "selected_myco" for row in copied),
        "n_total": len(samples),
        "samples": copied,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return samples
