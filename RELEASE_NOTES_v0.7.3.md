# mesa-cfdna v0.7.3

This release adds fixed probability blending and makes pandas preprocessing metadata-safe for downstream selectors that rely on sample IDs.

## Highlights

- Added `integration_method="probability_blend"` for binary classification.
- Added DataFrame-preserving normalization and variance filtering.
- Preserved sample IDs and feature names through optional normalization and variance filtering.
- Added stricter DataFrame feature-alignment checks at transform time.
- Expanded validation coverage for probability blending and DataFrame preprocessing.

## Compatibility

Existing `MESA_modality`, `MESA`, and `MESA_CV` calls are intended to remain backward compatible. NumPy preprocessing behavior is unchanged.

## PyPI publishing

Publishing GitHub Release `v0.7.3` triggers `.github/workflows/python-publish.yml`, which verifies the tag against `setup.py`, runs the test suite and smoke checks, builds the source/wheel distributions, validates them with `twine check`, and publishes through PyPI Trusted Publishing.

**Full changelog:** https://github.com/ChaorongC/mesa_cfdna/compare/v0.7.2...v0.7.3
