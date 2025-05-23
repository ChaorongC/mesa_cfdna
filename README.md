# Multimodal Epigenetic Sequencing Analysis (MESA)

---

## Overview

MESA(Multimodal Epigenetic Sequencing Analysis) is a flexible and sensitive method of capturing and integrating multimodal epigenetic information of cfDNA using a single experimental assay. This package is developed with sklearn-compatible API, integrating steps of preprocessing, scaling, feature selection, model training and cross-validation.

## Installation

```
# install package with pip
pip install mesa-cfdna
```

## Usage

The MESA package consists of three major components: 1. MESA single modality and multimodal integration, 2. MESA modality evaluation and selection, 3. MESA cross-validation

### Single modality

## Example

Check the Jupyter notebook `demo.ipynb` for a tutorial on how to run MESA.

#### Parameters

```shell
MESA_modality(
        top_n=100,
        variance_threshold=0,
        normalization=False,
        missing=0.1,
	classifier=RandomForestClassifier(random_state=0, n_jobs=-1),
	selector=GenericUnivariateSelect(score_func=wilcoxon, mode="k_best", param=2000),
        boruta_estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
	random_state=0)
```

**top_n : *int, default=100***

> The number of features to select for modality.

**variance_threshold** :***int or float, default=0***

> The threshold for the minimal variance(after normalization if `normalization`=True) allowed for features. Features with varaince < `variance_threshold` will be dropped.

**normalization:** ***bool, default=True***

> Whether to perform normalization(l2) before feature selection and model training.

**classifier : *estimator/model implementing ‘fit’ and 'predict_proba', default: `RandomForestClassifier(random_state=0, n_jobs=-1)`***

> A model/estimator trains on the final selected feature subset.

__selector__ :* **int or GenericUnivariateSelect or SequentialFeatureSelector, default=GenericUnivariateSelect( score_func=wilcoxon, mode="k_best", param=2000)***

> * If int, then use `GenericUnivariateSelect(score_func=wilcoxon, mode="k_best", param=selector)` to select top `selector`
> * If `GenericUnivariateSelect` or `SequentialFeatureSelector`, use the `selector` to select first-step feature subset.

__missing__ : float, default=0.1

> Maximum % of missing values allowed for features.

__random_state__ : int, RandomState instance or None, default=0

> Controls the pseudo random number generation for shuffling the data.

**boruta_estimator**: estimator/model with ‘`fit`’ implement and '`feature_importances_`' attribute, default=`RandomForestClassifier(random_state=0, n_jobs=-1)`

> A supervised learning estimator, with a 'fit' method that returns the feature_importances_ attribute. Important features must correspond to high absolute values in the feature_importances_.

```shell
MESA(X_list, 
                  y, 
                  feature_selected, 
                  classifiers)
```

__X__ : list of dataframes of shape (n_features, n_samples)

> Input samples.
> A matrix containing features as rows with samples as columns.

__y__ : array-like of shape (n_samples,)

> Target values/labels/stages. Usually, we use 0 and 1 for 'normal/negative' and 'cancer/positive' samples.

__feature_selected__ :  list of tuples (n_samples)

> Features selected for each LOO iteration (same order with X)

__classifiers__ : a list of estimator object/model implementing ‘fit’ and 'predict_proba'

> The object to use to evalutate on test set at the end.

## Authors

- Yumei Li (yumei.li@uci.edu)
- JianFeng Xu (Jianfeng@heliohealth.com)
- Chaorong Chen (chaoronc@uci.edu)
- Wei Li (wei.li@uci.edu)
