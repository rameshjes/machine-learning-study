import pandas as pd

from sklearn import ensemble
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from scipy import sparse

"""
Here we will see performance of RF 
on the similar dataset (used for logreg.)
Note that: we don't need to normalize/standarize features
for tree-based algos.
"""

def train_rf_model(fold):
    
    df = pd.read_csv("../../data/cat_train_folds.csv", sep=",")

    features = [feature for feature in df.columns if feature not in ["id", "target", "kfold"]]

    # fill all nan with None
    # convert all feature to string type as all are categorical
    for feature in features:
        df.loc[:, feature] = df[feature].astype(str).fillna("NONE")

    for feature in features:
        le = preprocessing.LabelEncoder()
        le.fit(df[feature])
        df.loc[:, feature] = le.transform(df[feature])

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)

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

"""
We can also try to create one hot encoding of features,
convert them to sparse matrix, apply SVD and trainig RF over
that.
"""
def train_rf_svd(fold):

    df = pd.read_csv("../../data/cat_train_folds.csv", sep=",")
    features = [feature for feature in df.columns if feature not in ["id", "target", "kfold"]]
    
    # fill all nan with None
    # convert all feature to string type as all are categorical
    for feature in features:
        df.loc[:, feature] = df[feature].astype(str).fillna("NONE")

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)
    print(f"x train {df_train.shape}")
    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    print(f"full data {full_data.shape}")
    # initialize OneHotEncoder
    ohe = preprocessing.OneHotEncoder(sparse=True)
    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features])

    print(f"x train after transforming {x_train}")
    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize Truncated SVD
    # we arre reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)


    # fit svd on sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # trainform sparse training and valid data
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)
    print(f"x train after svd {x_train.shape}")

    rf_svd_model = ensemble.RandomForestClassifier(n_jobs=-1)
    rf_svd_model.fit(x_train, df_train.target.values)

    # validation
    valid_preds = rf_svd_model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"fold : {fold} auc : {auc}")

    return auc


    
if __name__ == "__main__":

    average_auc_score = 0
    total_folds = 5

    # for fold in range(total_folds):
    #     average_auc_score += train_rf_model(fold=fold)

    # print(f"Average auc score {average_auc_score/total_folds}")

    # print(f"Note that: folds are take much longer than as compared to the LRE")
    # print(f"Also, inference time of random forest is much longer and it takes much larger space")

    # print(f"lets try to reduce total no. of features by using singular value decomposition")

    average_svd_auc_score = 0

    for fold in range(total_folds):
        average_svd_auc_score += train_rf_svd(fold=fold)

    print(f"Average auc score using svd {average_svd_auc_score/total_folds}")
