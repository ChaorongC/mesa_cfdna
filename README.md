# Multimodal Epigenetic Sequencing Analysis (MESA)

A flexible and sensitive method for capturing and integrating multimodal epigenetic information from cell-free DNA (cfDNA) using a single experimental assay.

## Overview

MESA (Multimodal Epigenetic Sequencing Analysis) provides a comprehensive framework for analyzing multimodal epigenetic data from cfDNA. The package features a sklearn-compatible API that seamlessly integrates preprocessing, scaling, feature selection, model training, and cross-validation workflows.

### Key Features

- **Multimodal Integration**: Combine multiple epigenetic data modalities using ensemble stacking
- **Advanced Feature Selection**: Boruta algorithm combined with univariate selection to keep a balance between computation time and biomarker discovery
- **Robust Cross-Validation**: Built-in evaluation framework with performance metrics for easy finetuning
- **Flexible Pipeline**: Customizable preprocessing and classification components
- **Missing Value Handling**: Intelligent filtering and imputation strategies

## Installation

```bash
# Install package with pip
pip install mesa-cfdna
```

## Quick Start

```python
from mesa import MESA_modality, MESA, MESA_CV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your data
X_train, y_train = load_data()  # Your data loading function

# Single modality analysis
modality_1 = MESA_modality(top_n=50, classifier=RandomForestClassifier(random_state=0), variance_threshold=0, normalization=True)
modality_1.fit(X_train, y_train)
predictions = modality_1.transform_predict_proba(X_test)

modality_2 = MESA_modality(top_n=100, classifier=LogisticRegression(random_state=0), variance_threshold=0, normalization=False, missing=0)
modality_2.fit(X_train, y_train)
predictions = modality_2.transform_predict_proba(X_test)

# Multi-modality ensemble
modalities = [modality_1, modality_2]
mesa = MESA(modalities)
mesa.fit([X1_train, X2_train], y_train)
mesa_predictions = mesa.predict_proba([X1_test, X2_test])
```

## API Reference

### MESA_modality

Single modality analysis with comprehensive preprocessing and feature selection pipeline.

#### Parameters

| Parameter              | Type          | Default                   | Description                                              |
| ---------------------- | ------------- | ------------------------- | -------------------------------------------------------- |
| `top_n`              | int           | 100                       | Number of features to select using Boruta algorithm      |
| `variance_threshold` | float         | 0                         | Minimum variance threshold for feature filtering         |
| `normalization`      | bool          | False                     | Whether to apply L2 normalization                        |
| `missing`            | float         | 0.1                       | Maximum proportion of missing values allowed per feature |
| `classifier`         | estimator     | RandomForestClassifier()  | Final classifier for predictions                         |
| `selector`           | int/estimator | GenericUnivariateSelect() | Univariate feature selector                              |
| `boruta_estimator`   | estimator     | RandomForestClassifier()  | Base estimator for Boruta selection                      |
| `random_state`       | int           | 0                         | Random seed for reproducibility                          |

#### Methods

- `fit(X, y)`: Fit the preprocessing pipeline and classifier
- `transform(X)`: Apply preprocessing pipeline only
- `predict(X)`: Predict class labels for preprocessed data
- `predict_proba(X)`: Predict class probabilities for preprocessed data
- `transform_predict(X)`: Apply pipeline and predict in one step
- `transform_predict_proba(X)`: Apply pipeline and predict probabilities
- `get_support(step=None)`: Get indices of selected features
- `get_params(deep=True)`: Get model parameters

### MESA

Multi-modality ensemble with stacking architecture for integrating multiple data types.

#### Parameters

| Parameter          | Type         | Default                   | Description                                 |
| ------------------ | ------------ | ------------------------- | ------------------------------------------- |
| `modalities`     | list         | Required                  | List of MESA_modality objects               |
| `meta_estimator` | estimator    | LogisticRegression()      | Meta-learner for ensemble combination       |
| `random_state`   | int          | 0                         | Random seed for reproducibility             |
| `cv`             | cv generator | RepeatedStratifiedKFold() | Cross-validation strategy for meta-features |

#### Methods

- `fit(X_list, y)`: Fit all modalities and meta-estimator
- `predict(X_list_test)`: Predict class labels using ensemble
- `predict_proba(X_list_test)`: Predict class probabilities using ensemble
- `get_support(step=None)`: Get feature support from all modalities

### MESA_CV

Cross-validation wrapper for performance evaluation of MESA models.

#### Parameters

| Parameter        | Type         | Default                     | Description                              |
| ---------------- | ------------ | --------------------------- | ---------------------------------------- |
| `modality`     | estimator    | Required                    | MESA_modality or MESA object to evaluate |
| `random_state` | int          | 0                           | Random seed for reproducibility          |
| `cv`           | cv generator | StratifiedKFold(n_splits=5) | Cross-validation strategy                |

#### Methods

- `fit(X, y)`: Perform cross-validation on provided data
- `get_performance()`: Calculate mean ROC AUC score across CV folds

#### Attributes

- `cv_result`: List of (y_pred, y_true) tuples from each CV fold
- `modality`: The fitted modality estimator being evaluated

## Usage Examples

### Example 1: Single Modality Analysis

```python
import pandas as pd
import numpy as np
from mesa import MESA_modality, MESA_CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Load single modality data
X = pd.read_csv('methylation_data.csv', index_col=0)
y = pd.read_csv('labels.csv', index_col=0).values.ravel()

# Create and configure modality
modality = MESA_modality(
    top_n=50,
    missing=0.2,
    normalization=True,
    classifier=RandomForestClassifier(n_estimators=100, random_state=42)
)

# Fit the modality
modality.fit(X, y)

# Make predictions on new data
X_test = pd.read_csv('test_data.csv', index_col=0)
predictions = modality.transform_predict_proba(X_test)
print(f"Prediction probabilities shape: {predictions.shape}")

# Get selected features
selected_features = modality.get_support()
print(f"Number of selected features: {len(selected_features)}")

# Cross-validation evaluation
cv_eval = MESA_CV(
    modality=MESA_modality(top_n=50, missing=0.2),
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
)
cv_eval.fit(X, y)
auc_score = cv_eval.get_performance()
print(f"Cross-validation AUC: {auc_score:.3f}")
```

### Example 2: Multi-Modality Ensemble

```python
import pandas as pd
from mesa import MESA_modality, MESA, MESA_CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load multi-modal data
methylation_data = pd.read_csv('methylation.csv', index_col=0)
histone_data = pd.read_csv('histone_marks.csv', index_col=0)
chromatin_data = pd.read_csv('chromatin_accessibility.csv', index_col=0)
y = pd.read_csv('labels.csv', index_col=0).values.ravel()

# Define modality-specific configurations
modalities = [
    MESA_modality(
        top_n=100,
        missing=0.1,
        classifier=RandomForestClassifier(n_estimators=200, random_state=42),
        normalization=True
    ),
    MESA_modality(
        top_n=80,
        missing=0.15,
        classifier=SVC(probability=True, random_state=42),
        normalization=False
    ),
    MESA_modality(
        top_n=60,
        missing=0.2,
        classifier=LogisticRegression(random_state=42),
        normalization=True
    )
]

# Create MESA ensemble
mesa = MESA(
    modalities=modalities,
    meta_estimator=LogisticRegression(random_state=42),
    random_state=42
)

# Fit the ensemble
X_list = [methylation_data, histone_data, chromatin_data]
mesa.fit(X_list, y)

# Make ensemble predictions
X_test_list = [
    pd.read_csv('methylation_test.csv', index_col=0),
    pd.read_csv('histone_test.csv', index_col=0),
    pd.read_csv('chromatin_test.csv', index_col=0)
]
ensemble_predictions = mesa.predict_proba(X_test_list)
print(f"Ensemble predictions shape: {ensemble_predictions.shape}")

# Get feature support from all modalities
feature_supports = mesa.get_support()
for i, support in enumerate(feature_supports):
    print(f"Modality {i+1}: {len(support)} features selected")
```

### Example 3: Cross-Validation for Multi-Modality

```python
from mesa import MESA, MESA_modality, MESA_CV
from sklearn.model_selection import RepeatedStratifiedKFold

# Define ensemble for CV evaluation
modalities = [
    MESA_modality(top_n=50, missing=0.1),
    MESA_modality(top_n=40, missing=0.15),
    MESA_modality(top_n=60, missing=0.2)
]

mesa_ensemble = MESA(
    modalities=modalities,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
)

# Cross-validation evaluation
cv_eval = MESA_CV(
    modality=mesa_ensemble,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Perform CV on multi-modal data
X_list = [methylation_data, histone_data, chromatin_data]
cv_eval.fit(X_list, y)

# Get performance metrics
mean_auc = cv_eval.get_performance()
print(f"Multi-modal ensemble CV AUC: {mean_auc:.3f}")

# Access individual fold results
for i, (y_pred, y_true) in enumerate(cv_eval.cv_result):
    fold_auc = roc_auc_score(y_true, y_pred[:, 1])
    print(f"Fold {i+1} AUC: {fold_auc:.3f}")
```

### Example 4: Custom Feature Selection Pipeline

```python
from mesa import MESA_modality
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

# Custom modality with different feature selection
custom_modality = MESA_modality(
    top_n=30,
    variance_threshold=0.01,
    missing=0.05,
    normalization=True,
    selector=SelectKBest(score_func=f_classif, k=1000),
    classifier=GradientBoostingClassifier(n_estimators=100, random_state=42),
    boruta_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42)
)

# Fit and evaluate
custom_modality.fit(X, y)
custom_predictions = custom_modality.transform_predict_proba(X_test)

# Compare with default configuration
default_modality = MESA_modality()
cv_custom = MESA_CV(custom_modality)
cv_default = MESA_CV(default_modality)

cv_custom.fit(X, y)
cv_default.fit(X, y)

print(f"Custom configuration AUC: {cv_custom.get_performance():.3f}")
print(f"Default configuration AUC: {cv_default.get_performance():.3f}")
```

### Example 5: Feature Importance Analysis

```python
# Analyze feature importance across modalities
modality = MESA_modality(top_n=100)
modality.fit(X, y)

# Get feature support at different pipeline steps
missing_support = modality.get_support(step=0)  # After missing value filtering
variance_support = modality.get_support(step=1)  # After variance filtering
univariate_support = modality.get_support(step=2)  # After univariate selection
final_support = modality.get_support()  # Final selected features

print(f"Features after missing value filter: {len(missing_support)}")
print(f"Features after variance filter: {len(variance_support)}")
print(f"Features after univariate selection: {len(univariate_support)}")
print(f"Final selected features: {len(final_support)}")

# Get feature names if using DataFrame
if hasattr(X, 'columns'):
    selected_feature_names = X.columns[final_support]
    print(f"Selected features: {selected_feature_names.tolist()}")
```

## Performance Tips

1. **Memory Management**: For large datasets, consider reducing `top_n` and using `n_jobs=1` for Boruta
2. **Feature Selection**: Adjust `missing` threshold based on data quality
3. **Cross-Validation**: Use fewer repeats for initial exploration, more for final evaluation
4. **Ensemble Size**: Start with 2-3 modalities, add more based on performance gains

## Citation

If you use MESA in your research, please cite:

Li, Y., Xu, J., Chen, C. et al. Multimodal epigenetic sequencing analysis (MESA) of cell-free DNA for non-invasive colorectal cancer detection. Genome Med 16, 9 (2024). https://doi.org/10.1186/s13073-023-01280-6

## Authors

- **Chaorong Chen** - *Lead Developer* - c.chen@uci.edu
- **Wei Li** - *Principal Investigator* - wei.li@uci.edu

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions and support:

- Open an issue on GitHub
- Email: c.chen@uci.edu
- Documentation: [Link to full documentation]

---

**Keywords**: cfDNA, epigenetics, multimodal analysis, machine learning, feature selection, ensemble learning, stacking, bioinformatics, biomarker discovery, methylation, computational biology, early detection
