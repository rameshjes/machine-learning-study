import itertools
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics

"""
Here we will see performance of 
XGBoost (One of the most popular gradient boosting algorithms)
on the similar dataset (used for logreg.)
Note that: we don't need to normalize/standarize features
for tree-based algos.
"""


def train_xgb_model(fold):
    
    df = pd.read_csv("../../data/adult_folds.csv", sep=",")

    # print(df[df.target.values == 1].shape)
    # print(df[df.target.values == 0].shape)
    
    include_numerical_features = True

    num_cols = [
            "fnlwgt",
            "age",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
        ]
    
    add_more_features = True
    # if flag is False, then all num. features are removed
    if not include_numerical_features:
        # drop numerical cols
        df = df.drop(num_cols, axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # Create more features (using existing categorical features)
    cat_cols = [feature for feature in df.columns if feature 
                                not in ["kfold", "income"]
                                and feature not in num_cols]

    features = [feature for feature in df.columns if feature
                            not in ["kfold", "income"]]
    
    if add_more_features:
        df = feature_engineering(df, cat_cols)
   
    # fill all Nan values with None
    # Converting all cols to str
    # it does not matter because all are categories
    for feature in features:
        # skip numercal
        if feature not in num_cols:
            df.loc[:, feature] = df[feature].astype(str).fillna("NONE")
    
    for feature in features:
        if feature not in num_cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[feature])
            df.loc[:, feature] = le.transform(df[feature])

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    xgb_model = xgb.XGBClassifier(n_jobs=-1)
    
    xgb_model.fit(x_train, df_train.income.values)

    # validation
    valid_preds = xgb_model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"fold : {fold} auc : {auc}")

    return auc


def feature_engineering(df, cat_cols):

    """
    This function is used for feature engineering
    : param df: pandas dataframe with train/test data
    : param cat_cols: list of categorical columns
    : return: dataframe with new features
    """
    # This will create all 2-combinations of values in list
    # E.g. list(itertools.combinations([1,2,3], 2)) returns:
            # [(1,2), (1,3), (2,3)]

    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = \
                        df[c1].astype(str) + "_" + \
                        df[c2].astype(str)

    print(df.head())
    return df


if __name__ == "__main__":
    
    average_auc_score = 0
    total_folds = 5
    for fold in range(total_folds):
        average_auc_score += train_xgb_model(fold=fold)

    print(f"Average auc score {average_auc_score/total_folds}")