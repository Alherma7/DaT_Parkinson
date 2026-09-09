"""Stochastic, training-only data augmentation for DatCNN volumes.

Never used in data.py's deterministic preprocessing (shared between
train and inference) or at inference/validation time -- these
transforms are randomized per call, on purpose, and only make sense
inside a training DataLoader's `transform`.

Array shape throughout is `(channel, L-R, A-P, S-I)`, matching
`data.load_volume`'s `(1, *config.TARGET_SHAPE)` output (RAS
convention, `config.py`).
"""

import numpy as np
from scipy import ndimage

L_R_AXIS = 1


def random_flip(volume, rng, axis=L_R_AXIS, p=0.5):
    """Flips `volume` along `axis` (default the L-R axis -- the same
    asymmetry axis `features.py::signed_asymmetry` uses) with
    probability `p`. README.md's EDA section 6b cleared L-R flip as
    statistically safe (doesn't corrupt the asymmetry signal); this
    augmentation is the actual experiment testing whether it helps the
    CNN's log loss.
    """
    if rng.random() < p:
        return np.flip(volume, axis=axis).copy()
    return volume


def random_rotation(volume, rng, max_degrees=10.0, axes=(2, 3)):
    """Rotates `volume` by a random angle in `[-max_degrees, max_degrees]`
    within the `axes` plane. Default `axes=(2, 3)` (A-P, S-I) rotates
    around the L-R axis without mixing it into the rotated plane, so
    laterality stays meaningful under this augmentation -- `random_flip`
    above is the only transform allowed to touch the L-R axis.
    `order=1` (linear interpolation, no ringing), `reshape=False` (keeps
    input shape), `mode="nearest"` (no wrap-around at the volume edge).
    """
    angle = rng.uniform(-max_degrees, max_degrees)
    return ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1, mode="nearest")


def random_brightness_jitter(volume, rng, max_jitter=0.1):
    """Multiplies `volume` by a random factor in
    `[1 - max_jitter, 1 + max_jitter]`. Volumes are already z-score
    normalized (`data.py`), so this default is a much narrower range
    than the literature's raw-intensity jitter (e.g. Kurmi et al.'s
    `[0.1, 1.5]`, RESOURCES.md) -- a wide multiplicative jitter on
    already-standardized data would swamp the signal.
    """
    factor = rng.uniform(1 - max_jitter, 1 + max_jitter)
    return volume * factor


def augment_volume(volume, rng, flip_p=0.5, max_rotation_deg=10.0, max_brightness_jitter=0.1):
    """Composes the three augmentations above, in order: flip, rotate,
    brightness jitter. `rng` (a `numpy.random.Generator`) is passed in,
    not created here, so callers control reproducibility -- e.g. one
    `Generator` per `DatParkinsonDataset` instance.
    """
    volume = random_flip(volume, rng, p=flip_p)
    volume = random_rotation(volume, rng, max_degrees=max_rotation_deg)
    volume = random_brightness_jitter(volume, rng, max_jitter=max_brightness_jitter)
    return volume.astype(np.float32)
