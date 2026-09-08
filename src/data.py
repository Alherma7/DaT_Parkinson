"""Deterministic geometry/intensity preprocessing for DaT-SPECT volumes
(the CNN track). Every function here is the same one used at training and
inference time -- never reimplemented in the inference path
(`deep-learning-imaging.md`'s "Common mistakes").

Unlike `features.py` (the classical baseline's adaptive per-volume
striatum mask), this pipeline resamples to a fixed physical spacing and
crops/pads to a fixed physical box every time -- the standard pattern for
feeding a CNN a consistent input geometry.
"""

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np


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
