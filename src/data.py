"""Deterministic geometry/intensity preprocessing for DaT-SPECT volumes
(the CNN track). Every function here is the same one used at training and
inference time -- never reimplemented in the inference path
(`deep-learning-imaging.md`'s "Common mistakes").

Always resamples to a fixed physical spacing and crops/pads to a fixed
physical box shape -- the standard pattern for feeding a CNN a consistent
input geometry. The crop CENTER can be fixed (`config.CROP_CENTER_MM`,
every subject the same point -- the original design) or adaptive
(`load_volume(uid, center_mm="auto")`, each subject's own measured
striatum centroid via `features.striatum_center_mm` -- see project memory's
striatum-crop-coverage investigation for why the fixed-center design left
~29% of subjects' striatum partly outside the crop).
"""

import warnings

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_nl_means

import config
import features


def resample_to_spacing(volume, affine, target_spacing, order=1):
    """Resample `volume` (+ its affine) to `target_spacing` mm, through
    the full affine -- not just axis permutation. `nibabel.processing
    .resample_to_output` is nibabel's own affine-aware resampler, which is
    what EDA's finding requires (40% of volumes are oblique, up to 40.3
    deg; `as_closest_canonical()` does not correct this -- README.md).
    Its output is RAS-aligned, so no separate reorientation step is
    needed. `order=1` (trilinear) by default; medical volumes should not
    use the default `order=3` spline, which can ring/overshoot near sharp
    edges.

    Returns `(resampled_volume, resampled_affine)`.
    """
    img = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    out = nibproc.resample_to_output(img, voxel_sizes=target_spacing, order=order)
    return out.get_fdata(), out.affine


def crop_or_pad(volume, spacing, center_mm, target_shape):
    """Crop or zero-pad `volume` (already resampled to `spacing`) to
    `target_shape` voxels, centered at the volume's geometric center
    offset by `center_mm` (mm, RAS axes -- same convention as
    `config.CROP_CENTER_MM`). Zero-pads on any axis where `target_shape`
    extends past the volume's edge (the tight-FOV volumes noted in
    `config.py`: the crop fits `MIN_FOV_MM` by only 3mm on z).
    """
    volume = np.asarray(volume)
    out = np.zeros(target_shape, dtype=volume.dtype)
    src_slices, dst_slices = [], []
    for axis in range(3):
        center_vox = (volume.shape[axis] - 1) / 2.0 + center_mm[axis] / spacing[axis]
        start = int(round(center_vox - target_shape[axis] / 2.0))
        end = start + target_shape[axis]
        src_start, src_end = max(start, 0), min(end, volume.shape[axis])
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def normalize_intensity(volume, background_percentile=config.BACKGROUND_PERCENTILE,
                         background_max_fraction=config.BACKGROUND_MAX_FRACTION):
    """Background-aware per-volume z-score, computed on the *cropped*
    volume. `np.clip(volume, 0, None)` first (16 volumes are stored as
    int16, not uint16 -- EDA section 5, same guard as `features.py`).
    Background threshold = min(percentile(volume, background_percentile),
    background_max_fraction * volume.max()) -- the EDA-validated rule
    already in `config.py` (adaptive, since a flat percentile cuts into
    brain on tight-FOV volumes). z-score uses the foreground
    (above-threshold) voxels' mean/std, applied to the whole volume.
    """
    volume = np.clip(volume, 0, None).astype(np.float32)
    if volume.max() <= 0:
        return volume  # degenerate all-zero volume; nothing to normalize
    threshold = min(np.percentile(volume, background_percentile),
                     background_max_fraction * volume.max())
    foreground = volume[volume > threshold]
    if foreground.size == 0:
        foreground = volume.ravel()
    mean = foreground.mean()
    std = foreground.std()
    if std <= 0:
        std = 1.0
    return (volume - mean) / std


def _estimate_noise_sigma(volume):
    """Robust per-volume noise-sigma estimate: median absolute deviation
    of a Laplacian-filtered volume, scaled to a normal-distribution sigma
    (the standard MAD-to-sigma constant, 1/0.6745 -- Donoho & Johnstone
    1994's wavelet-domain estimator uses the same constant on a detail
    coefficient; a Laplace high-pass filter is the dependency-free
    equivalent here).

    Deliberately NOT `skimage.restoration.estimate_sigma`: that function
    requires PyWavelets, which is not in the DrivenData runtime's
    `uv.lock` (checked 2026-09-09) -- using it here would crash
    `submission_src/main.py` if this experiment is ever promoted to
    production, since `data.py` is the exact code that runs there.
    """
    laplacian = ndimage.laplace(volume)
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    return float(mad / 0.6745)


def denoise_volume(volume, patch_size=3, patch_distance=5):
    """Patch-wise Non-Local-Means denoising (Boulkrinat et al. 2025,
    RESOURCES.md's preprocessing pipeline step 3), applied to the
    resampled+cropped volume, before intensity normalization.
    `_estimate_noise_sigma` picks the noise-strength parameter `h` per
    volume (no fixed constant -- noise scale varies by scanner family,
    same reasoning as `normalize_intensity`'s adaptive background
    threshold). `channel_axis=None` throughout -- a single-channel 3D
    volume, not a multi-channel image.
    """
    sigma_est = _estimate_noise_sigma(volume)
    return denoise_nl_means(volume, h=1.15 * sigma_est, patch_size=patch_size,
                             patch_distance=patch_distance, channel_axis=None,
                             fast_mode=True)


def _warn_if_degenerate(normalized):
    """Shared by `load_volume` and `load_slab`. Never include a uid in
    this message: both functions run inside `submission_src/main.py`
    against real competition test volumes, and the platform scans
    submission logs for per-sample test-set information and disqualifies
    on it -- a caller that already has the uid (every caller does, it's
    their own argument) can pair it with this warning itself if per-uid
    debugging is needed locally.
    """
    foreground_fraction = float(np.mean(np.abs(normalized) > 1e-6))
    if foreground_fraction < 0.01 or normalized.std() <= 1e-6:
        warnings.warn("degenerate/near-empty volume after preprocessing")


def load_volume(uid, center_mm=None):
    """Load, resample, crop, (optionally denoise,) and normalize the
    volume for `uid`. Returns a `(1, *config.TARGET_SHAPE)` float32
    array -- the single function both training and inference call, never
    reimplemented in the inference path. `config.USE_NLM_DENOISING`
    (default `False` -- rung-4 experiment 5, not yet gate-validated)
    gates the denoising step; flipping it is the only change needed to
    promote or revert the experiment, no separate code path to drift.

    `center_mm`: crop-center override, mm offset on RAS axes (same
    convention as `config.CROP_CENTER_MM`) -- lets a caller crop around a
    per-subject or alternative offset instead of a single fixed constant,
    without a second copy of this function to keep in sync with the real
    inference path. Three forms: `None` (default) -> `config.CROP_CENTER_MM`,
    the same point for every subject; `"auto"` -> this volume's OWN
    measured striatum centroid (`features.striatum_center_mm`, computed on
    the resampled volume), falling back to `config.CROP_CENTER_FALLBACK_MM`
    when the mask is degenerate (0/1362 in training, but inference must not
    crash on it) -- the per-subject-centered production mode; or an
    explicit `(x, y, z)` tuple, e.g. the striatum-crop-coverage
    experiments in notebooks/25-26.
    """
    path = config.NIFTI_DIR / f"{uid}.nii.gz"
    img = nib.load(str(path))
    resampled, _ = resample_to_spacing(img.get_fdata(), img.affine, config.TARGET_SPACING)

    if center_mm is None:
        center_mm = config.CROP_CENTER_MM
    elif center_mm == "auto":
        center_mm = features.striatum_center_mm(resampled, config.TARGET_SPACING)
        if center_mm is None:
            center_mm = config.CROP_CENTER_FALLBACK_MM

    cropped = crop_or_pad(resampled, config.TARGET_SPACING, center_mm,
                           config.TARGET_SHAPE)
    if config.USE_NLM_DENOISING:
        cropped = denoise_volume(cropped)
    normalized = normalize_intensity(cropped)
    _warn_if_degenerate(normalized)
    return normalized[np.newaxis, ...].astype(np.float32)


def load_slab(uid):
    """Load, resample, crop, and normalize a thick 2x2x12mm axial "slab"
    through the striatum for `uid` -- Wenzel et al. 2019's slab geometry
    (RESOURCES.md), a 2D analogue of `load_volume`'s 3D crop for
    `model.DatSlab2DCNN` (roadmap item 6, architecture diversity).
    Reuses the same `resample_to_spacing`/`crop_or_pad`/`normalize_intensity`
    building blocks as `load_volume`, only with slab-specific spacing/shape
    (`config.SLAB_TARGET_SPACING`/`SLAB_TARGET_SHAPE`) and the same
    striatum-centered `config.CROP_CENTER_MM` offset already validated for
    the 3D track -- no new center estimate needed. No denoising path (that
    experiment was rejected for the 3D track and never validated here).

    The 12mm slab thickness is built from `SLAB_TARGET_SHAPE[2]` (6)
    voxels at `SLAB_TARGET_SPACING[2]` (2.0mm) each, averaged together
    here, rather than a single 12mm-thick resampled voxel -- see
    `config.py`'s comment above `SLAB_TARGET_SPACING`: a single coarse
    voxel makes `crop_or_pad`'s nearest-voxel rounding (+-0.5 voxel, i.e.
    +-6mm at 12mm spacing) large enough to miss the striatum entirely,
    confirmed via a synthetic-marker diagnostic (2026-09-11) and consistent
    with the fold-0 sanity check in
    `notebooks/24_slab2d_architecture_diversity.ipynb` scoring barely
    above the base-rate baseline before this fix. Averaging 6 finer
    (2mm) voxels keeps the same 12mm total physical thickness while
    cutting that rounding error to +-1mm.

    Returns a `(1, SLAB_TARGET_SHAPE[0], SLAB_TARGET_SHAPE[1])` float32
    array: the S-I axis is averaged out and a leading channel dim is
    added, matching `load_volume`'s `(1, *shape)` convention.
    """
    path = config.NIFTI_DIR / f"{uid}.nii.gz"
    img = nib.load(str(path))
    resampled, _ = resample_to_spacing(img.get_fdata(), img.affine, config.SLAB_TARGET_SPACING)
    cropped = crop_or_pad(resampled, config.SLAB_TARGET_SPACING, config.CROP_CENTER_MM,
                           config.SLAB_TARGET_SHAPE)
    normalized = normalize_intensity(cropped)
    _warn_if_degenerate(normalized)
    return normalized.mean(axis=2)[np.newaxis, ...].astype(np.float32)
