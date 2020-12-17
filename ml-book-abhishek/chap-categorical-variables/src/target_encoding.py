import copy
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

"""
Target Encoding is another way of feature engineering

Target encoding is a technique in which you map 
each category in a given feature to its mean target value,
but this must always be done in a cross-validated manner.

It means that the first thing you do is create the folds, 
and then use these folds to create target encoding features
for different columns of the data in the same
way you fit and predict the model on folds.

So, if you have created 5 folds, you have to create
target encoding 5 times such that in the end, you
have encoding for variables in each fold which are
not derived from the same fold.

And then when you fit your model, you must use the same folds
again. 

Target encoding for unseen test data can be derived from the full
training data or can be an average of all the 5 folds

"""

def mean_target_encoding(data):

    # make a copy of dataframe
    df = copy.deepcopy(data)

    # list of numerical cols
    num_cols = [
            "fnlwgt",
            "age",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
    ]

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # all cols are features except kfold and income
    features = [feature for feature in df.columns if feature 
                                not in ["kfold", "income"]]

    cat_cols = [feature for feature in df.columns if feature
                        not in ["kfold", "income"] 
                        and  feature not in num_cols]
    
    # fill all Nan values with None
    # Converting all cols to str
    # it does not matter because all are categories
    for feature in features:
        # skip numercal
        if feature not in num_cols:
            df.loc[:, feature] = df[feature].astype(str).fillna("NONE")

    # encode categorical variables
    for feature in features:
        if feature not in num_cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[feature])
            df.loc[:, feature] = le.transform(df[feature])

    # a list to store 5 validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # for all feature columns, i.e. categorical cols
        for column in features:

            if column not in num_cols:
                # create dict of category: mean target
                # print(df_train.groupby(column)["income"].mean())
                mapping_dict = dict(
                    df_train.groupby(column)["income"].mean()
                )
                # print(f"mapping dict {mapping_dict}")

                # column_enc is the new column we have with mean encoding
                df_valid.loc[:, 
                    column + "_enc"
                ] = df_valid[column].map(mapping_dict)


        # drop cat_cols as new are created
        df_valid = df_valid.drop(cat_cols, axis=1)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)

    # create full dataframe again and return 
    encoded_df = pd.concat(encoded_dfs, axis=0)

    return encoded_df
    

def run(df, fold):
    # note that folds are same as before
    # get training data using folds

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)

    # all cols are features except kfold and income
    features = [feature for feature in df.columns if feature 
                                not in ["kfold", "income"]]

    # train and valid data
    x_train = df_train[features].values
    x_valid = df_valid[features].values


    # initialize xgboost model
    xgb_model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )

    # print(x_train.shape)
    # print(df_train.income.values)
    xgb_model.fit(x_train, df_train.income.values)

    # validation
    valid_preds = xgb_model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"fold : {fold} auc : {auc}")

    return auc


if __name__ == "__main__":
    
    # read data
    df = pd.read_csv("../../data/adult_folds.csv", sep=",")

    print(df.shape)
    # create mean target encoded categories
    df = mean_target_encoding(df)
    # print(f"df columsn {df.columns}")
    # print(f"education_enc {df.education_enc.unique}")
    # print(f"shape after mean target encoding {df.shape}")    
    average_auc_score = 0
    total_folds = 5
    # run training and validation on 5 folds
    for fold_ in range(total_folds):
        average_auc_score += run(df, fold_)


    print(f"Average auc score {average_auc_score/total_folds}")