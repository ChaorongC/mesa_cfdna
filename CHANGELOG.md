# Changelog

All notable changes to `mesa-cfdna` are documented here.

## [0.7.3] - 2026-09-07

### Added

- Added `integration_method="probability_blend"` for binary-classification ensembles. It combines modality-level positive-class probabilities using equal or user-specified fixed weights without fitting a meta-estimator.
- Added `DataFrameVarianceThreshold`, which preserves pandas sample IDs and feature names while retaining standard scikit-learn variance-filtering behavior.
- Added `DataFrameNormalizer`, which preserves pandas sample IDs and feature names while retaining standard scikit-learn row-normalization behavior.
- Added validation coverage for probability blending and DataFrame-preserving preprocessing.

### Changed

- `MESA_modality` now uses DataFrame-preserving normalization and variance filtering internally, so pandas indices survive preprocessing and can be used safely by downstream selectors that need sample metadata.
- DataFrame preprocessing aligns incoming columns to the fitted training-column order and rejects missing or duplicated training features rather than silently changing feature semantics.
- Updated README documentation and pipeline figures to describe the available multimodal integration strategies.

### Compatibility

- No breaking public API change is intended for existing `MESA_modality`, `MESA`, or `MESA_CV` calls.
- NumPy inputs retain standard scikit-learn behavior for normalization and variance filtering.
- The distribution version is `0.7.3`; the previous PyPI release was `0.7.2`.

[0.7.3]: https://github.com/ChaorongC/mesa_cfdna/compare/v0.7.2...v0.7.3
