import pandas as pd

from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

"""
Here we will see performance of RF 
on the similar dataset (used for logreg.)
Note that: we don't need to normalize/standarize features
for tree-based algos.
"""

def run(fold):
    
    df = pd.read_csv("../../data/cat_train_folds.csv", sep=",")

    # print(df[df.target.values == 1].shape)
    # print(df[df.target.values == 0].shape)
    
    features = [feature for feature in df.columns if feature not in ["id", "target", "kfold"]]


    # fill all nan with None
    # convert all feature to string type as all are categorical

    for feature in features:
        df.loc[:, feature] = df[feature].astype(str).fillna("NONE")

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    print(f"df train type : {type(df_train)}")
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)

    print(f"df train : {df_train.shape}")
    print(f"df valid : {df_valid.shape}")

    for feature in features:
        
        le = preprocessing.LabelEncoder()
        le.fit(df[feature])

        df_train[feature] = le.transform(df_train[feature])
        df_valid[feature] = le.transform(df_valid[feature])
        # df_valid[:, feature] = le.transform(df_valid[feature])

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    rf_model = ensemble.RandomForestClassifier(n_jobs=-1)
    rf_model.fit(x_train, df_train.target.values)

    # validation
    valid_preds = rf_model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"fold : {fold} auc : {auc}")

    return auc

    
if __name__ == "__main__":

    average_auc_score = 0
    total_folds = 5
    for fold in range(total_folds):
        average_auc_score += run(fold=fold)

    print(f"Average auc score {average_auc_score/total_folds}")