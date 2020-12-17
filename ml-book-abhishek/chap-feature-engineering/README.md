# Feature Engineering

* Feature Engineering is not just about creating new features from data but also includes different types of normalization and transformations.

### Creating New Features

* **Features can be created based on mean/count/median, polynomial etc**
* Polynomial function from sklearn can be used to create features based on input polynomial degree
* Another way to create feature in **binning**. This will convert numerical values to categorical. (Uses `cut` function from pandas). In other words, it enables us to treat numerical features as categorical.


### Handling Missing Numerical Values
 Different ways can be considered:

* Filling them with 0s (not efficient)
* Filling them with mean value of all the values in columns
* K-nearest neighbour (Like use of KNNimputer function)
* Another way of imputing missing values would be to train a regression model that tries to predict missing values in a column based on other column
* * So, you can start with one column that has a missing value and treat this column as the target column for regression model without the missing values. 
* * Using all the other columns, you now train a model on samples from which there is no missing value in the concerned column and then try to predict target(the same column) for the samples that were removed earlier. 
* * This way, you get a more robust model based imputation

* **Always remember that imputing(replacing missing values) values for tree-based models is unnecessary as they can handle it themselves**

**Always remmeber to scale or normalize your features if you are using linear models like log. reg. or a model like SVM**

**No need of normalization for tree-based methods**


## Converting Continuous/Numerical features to Categories

* This is done by **binning**
* **Binning** enables you treat numerical features to categorical
* To do this, `cut` function by pandas is used
* * `pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')`
* * Bin values into discrete intervals.
* * Use `cut` when you need to segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable.

#### Log tranformation

The log transformation is, arguably, the most popular among the different types of transformations used to transform skewed data to approximately conform to normality.

If the original data follows a log-normal distribution or approximately so, then the log-transformed data follows a normal or near normal distribution.


#### Identifying High Variance Features and Reduce their Variance

* To determine the variance of particular feature, execute
` df_feature_name.var() `
* To **reduce their variance**, apply log transformation or even exponential transformation can be used. To do this `df_feature_name.apply(lambda x: np.log(1+x)).var()` 