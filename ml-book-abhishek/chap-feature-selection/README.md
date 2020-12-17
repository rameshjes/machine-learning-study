# Feature Selection Chapter

Idea is to reduce the total no. of features. If we have lot of features, then we might also have a lot of training examples to capture all the features. 
Furthermore, having too many features pose a well known problem called as `curse of dimensionality`


## Ways of Feature Selection

### 1. Remove Features with Very Low Variance

* Simple form of selecting features would be to `remove features with very low variance`. E.g: If the features have a very low variance (i.e. very close to 0), they are close to being constant and thus, do not add any value to any model at all.
* sklearn provides functions for that i.e `from sklearn.feature_selection import VarianceThreshold`
* It **removes all features whose variance doesn’t meet some threshold**. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

### 2. Remove Features with High Correlation
* To calculate the correlation between different numerical features, **Pearson correlation** can be used
* pandas provides function for that i.e. `df.corr()`

Other ways of feature selection comes under `Univariate feature selection`.

### Univariate Feature Selection
* Univariate feature selection is way of computing score of each feature against a given target. 
* In other words, it selects the variable most related to the target outcome.
* Following are the some popular methods for univariate feature selection
* * For regression : `f_regression`, and `mutual_info_regression`
* * For classification : `chi2`, `f_classification`, and `mutual_info_classification`
* Two ways of using these in sklearn:
* * **SelectKBest:** Keeps the top-k scoring features
* * **SelectPercentile** Keeps the top features which are in a percentage specified by the user.

* Note that **chi2 can only be used for data which is non-negative in nature. E.g particularly useful feature selection technique in NLP when we have of BOW of TF-IDF features.**

* **Univariate feature selection may not always perform well**. Most of the time, people prefer dong feature selection using a machine learning model

### Feature Selection using ML Model
* One simple form of feature selection is called **Greedy or backward approach**
* Keep only the variables that you can remove from the learning process without damaging its performance.
* Following are the steps for greedy approach
* * First step is to choose a model
* * Second step is to select a loss/scoring function
* * Iteratively evaluate each feature and add it to the list of **good** features if it improves loss/score.
* This approach is computationally expensive, as all the time we are adding the features and training/evaluating the model.
* Check script `greedy.py` It returns scores and a list of feature indices

#### Another Greedy approach (Recursive Feature Elimination (RFE))

* In previous method, we started with one feature and kept adding new features, but in **RFE, we start with all features and keep removing one feature in every iteration that provides the least value to a given moel.**

## How do know which feature offers the least value?
* When we use **SVM** or **LR**, we get a coefficient for each feature that decides the importance of the feature.
* In case of **Tree based**, we get feature importance in place of coefficients.
* In each iteration, we can eliminate the least important feature and keep eliminating it until we reach the no. of features needed. (So yes, we can decide how many features we want to keep)


* When we are applying RFE, in each iteration, we remove the feature which has less importance or feature which has a coefficient close to 0.

* Note: **when we use LR. for binary classification, coefficients for features are more positive if they are important for positive class and more negative if they are important for negative class.