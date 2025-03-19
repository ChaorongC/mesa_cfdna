# MESA

<ins>M</ins>ultimodal <ins>E</ins>pigenetic <ins>S</ins>equencing <ins>A</ins>nalysis (MESA) is a flexible and sensitive method of capturing and integrating multimodal epigenetic information of cfDNA using a single experimental assay.
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2023-02-11 02:27:49e original MESA paper, please refer to this tutorial: https://rpubs.com/LiYumei/926228.

## Dependencies
- Python >=3.6
- deepTools
- bedtools
- DANPOS2
- BSMAP
- UCSC tools
- Python Package
  -  pandas
  -  numpy
  -  scikit-learn = 0.24.2
  -  joblib
  -  itertools
  -  boruta_py
  -  deep-forest

## Installation
Clone the repository with `git`:
```shell
git clone https://github.com/ChaorongC/MESA
cd MESA
```

Or download the repository with `wget`:
```shell
wget https://github.com/ChaorongC/MESA/archive/refs/heads/main.zip
unzip MESA-main.zip
cd MESA-main
```
## Usage
The Python script `MESA.py` in the root directory is the main program for MESA. 
The function `MESA_single()` in 'MESA.py' is for analysis on a single type of feature, and the function `MESA_integration()` is for combining results on different types of features and returning the multimodal prediction result.

## Example
Check the Jupyter notebook `demo.ipynb` for a tutorial on how to run MESA.

#### Parameters
```shell
MESA_single(X,
        y,
        estimator,
        classifiers=[],
        cv=5,
        random_state=0,
        min_feature=10,
        n_jobs=-1,
        scoring='roc_auc',
        boruta_top_n_feature=1000)
```
__X__ : dataframe of shape (n_features, n_samples)
  >Input samples.
  >A matrix containing features as rows with samples as columns.
   
__y__ : array-like of shape (n_samples,)
  >Target values/labels/stages. Usually, we use 0 and 1 for 'normal/negative' and 'cancer/positive' samples.

__estimator__ : estimator object/model implementing ‘fit’
  >The object used to fit the data.
  >A model that is used to evaluate feature subsets in each iteration of sequential backward selection.
    
__classifiers__ : a list of estimator object/model implementing ‘fit’ and 'predict_proba'
  >The object to use to evalutate on test set at the end.
  >A model used to train on the final selected feature subset then test on the testing set.

__cv__ : int, cross-validation generator or an iterable, default=5
  >(Adopted from `sklearn.model_selection.cross_val_score`) Determines the cross-validation splitting strategy. Possible inputs for cv are: 
  >None, to use the default 5-fold cross validation; int, to specify the number of folds in a (Stratified)KFold; CV splitter, An iterable yielding (train, test) splits as arrays of indices.
            
__random_state__ : int, RandomState instance or None, default=0
  >Controls the pseudo random number generation for shuffling the data.
  
__min_feature : int, default=10
  >The minimal feature size SBS should consider.
  
__n_jobs__ : int, default=-1
  >Number of jobs to run in parallel. When evaluating a new feature to add or remove, the cross-validation procedure is parallel over the folds. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.  
  
__scoring__ : str or callable, default='roc_auc'
  >For SBS process, a str (see scikit-learn model evaluation documentation) or a scorer callable object/function with signature scorer(estimator, X, y) which should return only a single value. Compatible with `sklearn.model_selection.cross_val_score`.    
    
__boruta_top_n_feature__ : int, default=1000
  >Features to select for SBS in the Boruta algorithm.
  >Features are first ranked by Boruta then output for SBS for further selection.
    
```shell
MESA_integration(X_list, 
                  y, 
                  feature_selected, 
                  classifiers)
```
__X__ : list of dataframes of shape (n_features, n_samples)
  >Input samples.
  >A matrix containing features as rows with samples as columns.
  
__y__ : array-like of shape (n_samples,)
  >Target values/labels/stages. Usually, we use 0 and 1 for 'normal/negative' and 'cancer/positive' samples.
  
__feature_selected__ :  list of tuples (n_samples) 
  >Features selected for each LOO iteration (same order with X)
  
__classifiers__ : a list of estimator object/model implementing ‘fit’ and 'predict_proba'
  >The object to use to evalutate on test set at the end.


## Authors
- Yumei Li (yumei.li@uci.edu)
- JianFeng Xu (Jianfeng@heliohealth.com)
- Chaorong Chen (chaoronc@uci.edu)
- Wei Li (wei.li@uci.edu)
