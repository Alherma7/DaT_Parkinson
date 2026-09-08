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
